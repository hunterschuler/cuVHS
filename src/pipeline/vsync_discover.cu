#include "pipeline/vsync_discover.h"
#include <cuda_runtime.h>

// Your exact kernel:
__global__ void k_discover_vsyncs(
    const double* d_demod_05,
    int* d_candidate_indices,
    int* d_candidate_count,
    int candidate_capacity,
    int total_samples,
    double threshold,
    int min_width,
    int max_width)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_samples) return;

    // Detect falling edge to start pulse measurement
    bool is_start = false;
    if (idx == 0) {
        is_start = (d_demod_05[idx] <= threshold);
    } else {
        is_start = (d_demod_05[idx] <= threshold && d_demod_05[idx - 1] > threshold);
    }

    if (is_start) {
        int width = 0;
        // Measure pulse width (bounded by max_width to prevent infinite loops)
        while (idx + width < total_samples && 
               d_demod_05[idx + width] <= threshold && 
               width <= max_width) 
        {
            width++;
        }
        
        // If the width matches a VSYNC serration (~27us)
        if (width >= min_width && width <= max_width) {
            int out_idx = atomicAdd(d_candidate_count, 1);
            if (out_idx < candidate_capacity) {
                d_candidate_indices[out_idx] = idx;
            }
        }
    }
}

// The Host Wrapper (This is what pipeline.cu actually calls)
void discover_vsyncs(const double* d_demod_05,
                     int* d_candidate_indices,
                     int* d_candidate_count,
                     int candidate_capacity,
                     int total_samples,
                     const VideoFormat& fmt)
{
    int threads = 256;
    int blocks = (total_samples + threads - 1) / threads;

    // ~20µs to 35µs window to catch VSYNC serrations.
    // We use a slightly wider window to account for FFT ringing on the boundaries.
    int min_vsync_samples = (int)(20.0e-6 * fmt.sample_rate);
    int max_vsync_samples = (int)(35.0e-6 * fmt.sample_rate);

    k_discover_vsyncs<<<blocks, threads>>>(
        d_demod_05, 
        d_candidate_indices, 
        d_candidate_count,
        candidate_capacity,
        total_samples, 
        fmt.pulse_threshold_hz,
        min_vsync_samples, 
        max_vsync_samples
    );
}
