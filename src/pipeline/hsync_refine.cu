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

// Find falling-edge zero crossing: first sample where signal drops below target.
// Returns fractional sample position via linear interpolation, or -1.0 if not found.
__device__ double find_falling_crossing(
    const double* sig,
    int start, int count,
    double target,
    int total_samples)
{
    if (start < 0 || start + count > total_samples || count < 2) return -1.0;

    for (int i = start + 1; i < start + count; i++) {
        double prev = sig[i - 1];
        double curr = sig[i];

        if (prev >= target && curr < target) {
            double a = prev - target;
            double b = curr - target;
            double frac = (a - b != 0.0) ? (a / (a - b)) : 0.0;
            return (double)(i - 1) + frac;
        }
    }
    return -1.0;
}

// Find rising-edge zero crossing: first sample where signal rises above target.
// Returns fractional sample position via linear interpolation, or -1.0 if not found.
// Matches Python calczc_do(..., edge=1): requires start sample to be below target.
__device__ double find_rising_crossing(
    const double* sig,
    int start, int count,
    double target,
    int total_samples)
{
    if (start < 0 || start + count > total_samples || count < 2) return -1.0;

    // Python: if data[start] > target, return NONE (already past threshold)
    if (sig[start] > target) return -1.0;

    for (int i = start + 1; i < start + count; i++) {
        double prev = sig[i - 1];
        double curr = sig[i];

        if (prev < target && curr >= target) {
            double a = target - prev;
            double b = target - curr;
            double frac = (a + b != 0.0) ? (a / (a - b)) : 0.0;
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
    int active_line_start,       // skip vblank lines
    double hsync_width_samples,  // ~4.7 µs in samples
    double right_edge_offset)    // 2.25 * (sample_rate_mhz / 40.0)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_lines = num_fields * lines_per_frame;
    if (idx >= total_lines) return;

    int field = idx / lines_per_frame;
    int line = idx % lines_per_frame;

    if (line < active_line_start) return;

    double* loc = &linelocs[field * lines_per_frame + line];
    double original = *loc;
    int one_usec = (int)(one_usec_samples + 0.5);

    // ---------------------------------------------------------------
    // Left-edge detection (falling edge = leading edge of hsync pulse)
    // ---------------------------------------------------------------
    int search_start = (int)(original - one_usec);
    int search_count = one_usec * 2;

    double zc = find_falling_crossing(demod_05, search_start, search_count,
                                      pulse_threshold_hz, total_demod_samples);

    double left_result = -1.0;

    if (zc >= 0.0) {
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

        if (porch_level > sync_level) {
            double midpoint = (sync_level + porch_level) / 2.0;

            int refine_start = (int)(original - one_usec * 2);
            int refine_count = (int)(one_usec_samples * 4);

            double zc2 = find_falling_crossing(demod_05, refine_start, refine_count,
                                               midpoint, total_demod_samples);

            if (zc2 >= 0.0 && fabs(zc2 - zc) <= one_usec_samples * 0.5) {
                left_result = zc2;
            }
        }
    }

    // ---------------------------------------------------------------
    // Right-edge detection (rising edge = trailing edge of hsync pulse)
    // Python: "less likely to be messed up by overshoot"
    //
    // Search for the rising edge near the expected end of hsync pulse.
    // From the right cross, derive the line start position:
    //   lineloc = right_cross - hsync_width + offset
    // ---------------------------------------------------------------
    double right_result = -1.0;

    // Search window: start at (original + hsync_width - 1µs)
    // Python: ll1 + normal_hsync_length - one_usec, count = normal_hsync_length * 2
    int right_search_start = (int)(original + hsync_width_samples - one_usec_samples);
    int right_search_count = (int)(hsync_width_samples * 2);

    double right_cross = find_rising_crossing(demod_05, right_search_start,
                                              right_search_count,
                                              pulse_threshold_hz,
                                              total_demod_samples);

    if (right_cross >= 0.0) {
        // Derive estimated leading edge from right cross
        double zc_fr = right_cross - hsync_width_samples;
        int zc_fr_int = (int)(zc_fr + 0.5);

        // Sync level: 1.0 to 2.5 µs after the derived leading edge
        // Python: demod_05[zc_fr + 1µs : zc_fr + 2.5µs]
        int r_sync_start = zc_fr_int + (int)(1.0 * one_usec_samples);
        int r_sync_count = (int)(1.5 * one_usec_samples);
        double r_sync_level = window_median(demod_05, r_sync_start, r_sync_count,
                                            total_demod_samples);

        // Porch level: hsync_width + 1µs to hsync_width + 2µs after derived leading edge
        // Python: demod_05[zc_fr + normal_hsync_length + 1µs : zc_fr + normal_hsync_length + 2µs]
        int r_porch_start = zc_fr_int + (int)(hsync_width_samples + 1.0 * one_usec_samples);
        int r_porch_count = (int)(1.0 * one_usec_samples);
        double r_porch_level = window_median(demod_05, r_porch_start, r_porch_count,
                                             total_demod_samples);

        if (r_porch_level > r_sync_level) {
            double r_midpoint = (r_sync_level + r_porch_level) / 2.0;

            // Refine: find rising edge at midpoint threshold
            // Python: calczc_do(demod_05, ll1 + normal_hsync_length - one_usec, midpoint, count=400)
            int r_refine_start = (int)(original + hsync_width_samples - one_usec_samples);
            int r_refine_count = (int)(hsync_width_samples * 2);

            double zc2_r = find_rising_crossing(demod_05, r_refine_start,
                                                r_refine_count, r_midpoint,
                                                total_demod_samples);

            if (zc2_r >= 0.0 && fabs(zc2_r - right_cross) <= one_usec_samples * 0.5) {
                // Derive lineloc from right edge
                // Python: right_cross - normal_hsync_length + (2.25 * (sample_rate_mhz / 40.0))
                double candidate = right_cross - hsync_width_samples + right_edge_offset;

                // Sanity: should be within ±2µs of left-edge result (or original if no left)
                double reference = (left_result >= 0.0) ? left_result : original;
                if (fabs(candidate - reference) <= one_usec_samples * 2.0) {
                    right_result = candidate;
                }
            }
        }
    }

    // ---------------------------------------------------------------
    // Result selection: prefer right-edge (Python behavior)
    // ---------------------------------------------------------------
    if (right_result >= 0.0) {
        *loc = right_result;
    } else if (left_result >= 0.0) {
        *loc = left_result;
    }
    // else: keep original
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
    double sample_rate_mhz = fmt.sample_rate / 1.0e6;
    double right_edge_offset = 2.25 * (sample_rate_mhz / 40.0);

    k_hsync_refine<<<blocks, threads>>>(
        d_demod_05,
        d_linelocs,
        num_fields,
        fmt.lines_per_frame,
        total_demod_samples,
        fmt.pulse_threshold_hz,
        one_usec_samples,
        fmt.active_line_start,
        fmt.hsync_width,
        right_edge_offset);
}

__global__ void k_hsync_refine_debug_analyze(
    const double* __restrict__ demod_05,
    const double* __restrict__ before,
    const double* __restrict__ after,
    int* __restrict__ large_delta_count,
    int* __restrict__ isolated_jump_count,
    int* __restrict__ refined_sync_like_count,
    int num_fields,
    int lines_per_frame,
    int total_demod_samples,
    int active_line_start,
    double pulse_threshold_hz,
    double large_delta_threshold,
    double isolated_delta_threshold)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_lines = num_fields * lines_per_frame;
    if (idx >= total_lines) return;

    int field = idx / lines_per_frame;
    int line = idx % lines_per_frame;
    if (line < active_line_start || line + 1 >= lines_per_frame) return;

    int base = field * lines_per_frame;
    double before_loc = before[base + line];
    double after_loc = after[base + line];
    double delta = after_loc - before_loc;

    if (fabs(delta) > large_delta_threshold) {
        atomicAdd(large_delta_count, 1);
    }

    if (line > active_line_start && line + 1 < lines_per_frame) {
        double prev_delta = after[base + line - 1] - before[base + line - 1];
        double next_delta = after[base + line + 1] - before[base + line + 1];
        double neighbor_avg = 0.5 * (prev_delta + next_delta);
        if (fabs(delta - neighbor_avg) > isolated_delta_threshold) {
            atomicAdd(isolated_jump_count, 1);
        }
    }

    int sample = (int)(after_loc + 0.5);
    if (sample >= 0 && sample < total_demod_samples && demod_05[sample] <= pulse_threshold_hz) {
        atomicAdd(refined_sync_like_count, 1);
    }
}

void hsync_refine_debug_analyze(const double* d_demod_05,
                                const double* d_linelocs_before,
                                const double* d_linelocs_after,
                                int* d_large_delta_count,
                                int* d_isolated_jump_count,
                                int* d_refined_sync_like_count,
                                int num_fields,
                                int total_demod_samples,
                                const VideoFormat& fmt)
{
    int total_lines = num_fields * fmt.lines_per_frame;
    int threads = 256;
    int blocks = (total_lines + threads - 1) / threads;
    double one_usec_samples = 1.0e-6 * fmt.sample_rate;

    k_hsync_refine_debug_analyze<<<blocks, threads>>>(
        d_demod_05,
        d_linelocs_before,
        d_linelocs_after,
        d_large_delta_count,
        d_isolated_jump_count,
        d_refined_sync_like_count,
        num_fields,
        fmt.lines_per_frame,
        total_demod_samples,
        fmt.active_line_start,
        fmt.pulse_threshold_hz,
        1.5 * one_usec_samples,
        1.0 * one_usec_samples);
}
