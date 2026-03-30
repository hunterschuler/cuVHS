#include "pipeline/sync_pulses.h"
#include <cuda_runtime.h>
#include <cstdio>

// One thread per field: sequential scan matching Python findpulses_numba_raw().
// Simple threshold crossing detector.
//
// If d_field_offsets is non-null, uses variable field offsets; otherwise assumes
// uniform stride of samples_per_field.
__global__ void k_find_pulses(
    const double* demod_05,      // contiguous demod output
    int* pulse_starts,           // [num_fields x MAX_PULSES]
    int* pulse_lengths,          // [num_fields x MAX_PULSES]
    int* pulse_count,            // [num_fields]
    const size_t* field_offsets, // optional: per-field offsets (or nullptr for uniform stride)
    int samples_per_field,
    double threshold,            // pulse_threshold_hz: samples <= this are "in pulse"
    int max_pulses,
    int num_fields)
{
    int field = blockIdx.x * blockDim.x + threadIdx.x;
    if (field >= num_fields) return;

    // Compute field offset: either from array or uniform stride
    size_t field_offset = field_offsets ? field_offsets[field] : (size_t)field * samples_per_field;
    
    const double* sig = demod_05 + field_offset;
    int* starts  = pulse_starts  + field * max_pulses;
    int* lengths = pulse_lengths + field * max_pulses;

    bool in_pulse = (sig[0] <= threshold);
    int cur_start = 0;
    int count = 0;

    for (int i = 0; i < samples_per_field; i++) {
        double val = sig[i];
        if (in_pulse) {
            if (val > threshold) {
                int length = i - cur_start;
                // Record pulse if length > 0 and not at sample 0
                // (Python: "if cur_start != 0" — skip pulse starting at pos 0)
                if (length > 0 && cur_start != 0 && count < max_pulses) {
                    starts[count]  = (int)(field_offset + cur_start);  // absolute position
                    lengths[count] = length;
                    count++;
                }
                in_pulse = false;
            }
        } else {
            if (val <= threshold) {
                cur_start = i;
                in_pulse = true;
            }
        }
    }

    pulse_count[field] = count;
}

void sync_pulses(const double* d_demod_05,
                 int* d_pulse_starts,
                 int* d_pulse_lengths,
                 int* d_pulse_count,
                 const size_t* d_field_offsets,  // optional: per-field offsets
                 int num_fields,
                 size_t samples_per_field,
                 const VideoFormat& fmt)
{
    int threads = 256;
    int blocks = (num_fields + threads - 1) / threads;

    k_find_pulses<<<blocks, threads>>>(
        d_demod_05,
        d_pulse_starts,
        d_pulse_lengths,
        d_pulse_count,
        d_field_offsets,  // can be nullptr for uniform stride
        (int)samples_per_field,
        fmt.pulse_threshold_hz,
        MAX_PULSES,
        num_fields);
}
