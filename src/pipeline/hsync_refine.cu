#include "pipeline/hsync_refine.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

// ============================================================================
// Kernel 4: Hsync Zero-Crossing Refinement
//
// For each line, refines the lineloc position using two-pass zero-crossing
// detection on the sync signal (demod_05, in Hz domain).
//
// Pass 1: Find initial crossing of pulse_threshold_hz
// Pass 2: Measure sync/porch levels, refine at midpoint
//
// Linear interpolation between samples gives sub-sample accuracy.
// Matches Python refine_linelocs_hsync() from vhsdecode/sync.pyx.
// ============================================================================

// Find zero crossing: first sample where signal crosses target level
// (looking for falling edge: signal goes from above to below target,
//  which is the leading edge of the hsync pulse in Hz domain)
//
// Returns fractional sample position via linear interpolation,
// or -1.0 if no crossing found.
__device__ double find_zero_crossing(
    const double* sig,
    int start, int count,
    double target,
    int total_samples)
{
    if (start < 0 || start + count > total_samples || count < 2) return -1.0;

    for (int i = start + 1; i < start + count; i++) {
        double prev = sig[i - 1];
        double curr = sig[i];

        // Falling edge: prev >= target, curr < target
        if (prev >= target && curr < target) {
            // Linear interpolation for sub-sample accuracy
            double a = prev - target;
            double b = curr - target;
            double frac = (a - b != 0.0) ? (a / (a - b)) : 0.0;
            return (double)(i - 1) + frac;
        }
    }
    return -1.0;
}

// Compute median of a small window (simple selection for small N)
__device__ double window_median(
    const double* sig,
    int start, int count,
    int total_samples)
{
    if (start < 0 || start + count > total_samples || count <= 0) return 0.0;

    // For small windows (<32 samples), bubble sort partial to find median
    // In practice these windows are 14-42 samples
    double buf[64];
    int n = min(count, 64);
    for (int i = 0; i < n; i++) buf[i] = sig[start + i];

    // Partial sort to find median (only need to sort half)
    int mid = n / 2;
    for (int i = 0; i <= mid; i++) {
        for (int j = i + 1; j < n; j++) {
            if (buf[j] < buf[i]) {
                double tmp = buf[i];
                buf[i] = buf[j];
                buf[j] = tmp;
            }
        }
    }
    return buf[mid];
}

__global__ void k_hsync_refine(
    const double* __restrict__ demod_05,   // sync signal (Hz domain)
    double* __restrict__ linelocs,          // [num_fields x lines_per_frame] in/out
    int num_fields,
    int lines_per_frame,
    int total_demod_samples,
    double pulse_threshold_hz,   // initial crossing threshold (-20 IRE in Hz)
    double one_usec_samples,     // 1 µs in samples
    int active_line_start)       // skip vblank lines
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_lines = num_fields * lines_per_frame;
    if (idx >= total_lines) return;

    int field = idx / lines_per_frame;
    int line = idx % lines_per_frame;

    // Skip vblank lines (VSYNC area has no regular hsync)
    // Lines 0-9 are vblank for NTSC, 0-7 for PAL
    if (line < active_line_start) return;

    double* loc = &linelocs[field * lines_per_frame + line];
    double original = *loc;
    int one_usec = (int)(one_usec_samples + 0.5);

    // ---------------------------------------------------------------
    // Pass 1: Find initial zero crossing at pulse_threshold_hz
    // Search window: 1 µs before to 1 µs after the estimated lineloc
    // ---------------------------------------------------------------
    int search_start = (int)(original - one_usec);
    int search_count = one_usec * 2;

    double zc = find_zero_crossing(demod_05, search_start, search_count,
                                   pulse_threshold_hz, total_demod_samples);

    if (zc < 0.0) return;  // no crossing found, keep original

    // ---------------------------------------------------------------
    // Pass 2: Measure sync and porch levels, refine at midpoint
    //
    // sync_level: median of signal 1.0-2.5 µs after crossing
    //            (in the middle of the sync pulse)
    // porch_level: median of signal 8-9 µs after crossing
    //            (the back porch / blanking level)
    // ---------------------------------------------------------------
    int zc_int = (int)(zc + 0.5);

    // Sync level: 1.0 to 2.5 µs after crossing
    int sync_start = zc_int + (int)(1.0 * one_usec_samples);
    int sync_count = (int)(1.5 * one_usec_samples);
    double sync_level = window_median(demod_05, sync_start, sync_count,
                                       total_demod_samples);

    // Porch level: 8 to 9 µs after crossing
    int porch_start = zc_int + (int)(8.0 * one_usec_samples);
    int porch_count = (int)(1.0 * one_usec_samples);
    double porch_level = window_median(demod_05, porch_start, porch_count,
                                        total_demod_samples);

    // Sanity check: porch should be above sync (higher Hz = higher IRE)
    if (porch_level <= sync_level) return;

    // Midpoint threshold
    double midpoint = (sync_level + porch_level) / 2.0;

    // ---------------------------------------------------------------
    // Pass 2 crossing: find refined zero crossing at midpoint level
    // Search in a wider window around the original estimate
    // ---------------------------------------------------------------
    int refine_start = (int)(original - one_usec * 2);
    int refine_count = (int)(one_usec_samples * 4);

    double zc2 = find_zero_crossing(demod_05, refine_start, refine_count,
                                    midpoint, total_demod_samples);

    if (zc2 < 0.0) return;

    // Sanity check: refined position should be within ±0.5 µs of initial
    if (fabs(zc2 - zc) > one_usec_samples * 0.5) return;

    // Accept the refined position
    *loc = zc2;
}


// ============================================================================
// Host entry point
// ============================================================================

void hsync_refine(const double* d_demod_05,
                  double* d_linelocs,
                  int num_fields,
                  int total_demod_samples,
                  const VideoFormat& fmt)
{
    int total_lines = num_fields * fmt.lines_per_frame;
    int threads = 256;
    int blocks = (total_lines + threads - 1) / threads;

    double one_usec_samples = 1.0e-6 * fmt.sample_rate;

    // Pass 1: Hsync zero-crossing refinement
    k_hsync_refine<<<blocks, threads>>>(
        d_demod_05,
        d_linelocs,
        num_fields,
        fmt.lines_per_frame,
        total_demod_samples,
        fmt.pulse_threshold_hz,
        one_usec_samples,
        fmt.active_line_start);

}
