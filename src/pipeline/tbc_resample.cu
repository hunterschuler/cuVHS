#include "pipeline/tbc_resample.h"
#include <cuda_runtime.h>
#include <cstdio>

// ============================================================================
// Kernel 5: TBC Resample
//
// Resamples the FM-demodulated video signal (in Hz domain) from the capture
// timebase to the fixed 4*fsc output timebase, producing uint16 TBC output.
//
// For each output pixel (field, line, col):
//   1. Map to input position via linelocs (linear interpolation between lines)
//   2. Catmull-Rom cubic interpolation of d_demod at that position
//   3. Convert Hz → IRE → uint16 (ld-tools compatible)
//
// One CUDA thread per output pixel.
// ============================================================================

__global__ void k_tbc_resample(
    const double* __restrict__ demod,       // [total_batch_samples] FM demod output (Hz)
    const double* __restrict__ linelocs,    // [num_fields x lines_per_frame]
    uint16_t* __restrict__ tbc_luma,        // [num_fields x output_field_lines x output_line_len]
    int* __restrict__ oob_pixel_count,      // optional debug counter
    int* __restrict__ bad_geom_line_count,  // optional debug counter
    int* __restrict__ sync_like_pixel_count,// optional debug counter
    int* __restrict__ sync_like_line_counts,// optional [num_fields x output_field_lines]
    int mark_fallback_white,                // optional debug visualization
    int mark_sync_like_white,               // optional debug visualization
    int num_fields,
    int lines_per_frame,
    int output_field_lines,
    int output_line_len,
    int active_line_start,          // first active video line in linelocs
    int active_video_start,         // first active output sample (exclude porch/sync)
    int active_video_end,           // one past last active output sample
    int total_demod_samples,        // total samples in d_demod buffer (for bounds check)
    double samples_per_line_nominal,
    double ire0,
    double hz_ire,
    double vsync_ire,
    double pulse_threshold_hz,
    double output_zero,
    double output_scale)
{
    // Thread index: one per output pixel
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int pixels_per_field = output_field_lines * output_line_len;
    int total_pixels = num_fields * pixels_per_field;
    if (idx >= total_pixels) return;

    // Decompose into field, line, column
    int field = idx / pixels_per_field;
    int rem = idx % pixels_per_field;
    int out_line = rem / output_line_len;
    int out_col = rem % output_line_len;

    // Map output row directly to the field's line index. The lineloc grid
    // already represents the full field; adding active_line_start here shifts
    // the raster down and wraps the top-of-field non-picture region to the
    // bottom of the 263-line output.
    int ll_line = out_line;
    int ll_next = ll_line + 1;

    // Bounds check on linelocs
    if (ll_next >= lines_per_frame) {
        tbc_luma[idx] = (uint16_t)output_zero;
        return;
    }

    // Get line start/end positions from linelocs (absolute positions in d_demod)
    double line_start = linelocs[field * lines_per_frame + ll_line];
    double line_end   = linelocs[field * lines_per_frame + ll_next];
    double line_len = line_end - line_start;

    // Count obviously suspicious geometry once per output line.
    if (out_col == 0 &&
        (line_len <= 0.0 ||
         line_len < 0.8 * samples_per_line_nominal ||
         line_len > 1.2 * samples_per_line_nominal)) {
        if (bad_geom_line_count) atomicAdd(bad_geom_line_count, 1);
    }

    // Compute input position via linear interpolation within the line
    double frac = (double)out_col / (double)output_line_len;
    double coord = line_start + frac * line_len;

    // Catmull-Rom cubic interpolation
    // Need 4 samples: p[0..3] centered around coord
    int ci = (int)coord;
    double x = coord - (double)ci;

    // Bounds check: need samples at ci-1, ci, ci+1, ci+2
    if (ci < 1 || ci + 2 >= total_demod_samples) {
        if (oob_pixel_count) atomicAdd(oob_pixel_count, 1);
        tbc_luma[idx] = mark_fallback_white ? 65535u : (uint16_t)output_zero;
        return;
    }

    double p0 = demod[ci - 1];
    double p1 = demod[ci];
    double p2 = demod[ci + 1];
    double p3 = demod[ci + 2];

    // Catmull-Rom formula: same as Python lddecode/utils.py scale()
    double a = p2 - p0;
    double b = 2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3;
    double c = 3.0 * (p1 - p2) + p3 - p0;
    double hz_value = p1 + 0.5 * x * (a + x * (b + x * c));

    // Convert Hz → IRE (shifted so vsync = 0)
    double ire = (hz_value - ire0) / hz_ire - vsync_ire;

    bool in_active_picture = (out_col >= active_video_start && out_col < active_video_end);
    bool sync_like = in_active_picture && (hz_value <= pulse_threshold_hz);
    if (sync_like) {
        if (sync_like_pixel_count) atomicAdd(sync_like_pixel_count, 1);
        if (sync_like_line_counts) {
            atomicAdd(&sync_like_line_counts[field * output_field_lines + out_line], 1);
        }
        if (mark_sync_like_white) {
            tbc_luma[idx] = 65535u;
            return;
        }
    }

    // Scale to uint16
    double out_val = ire * output_scale + output_zero;

    // Clamp and write
    if (out_val < 0.0) out_val = 0.0;
    if (out_val > 65535.0) out_val = 65535.0;
    tbc_luma[idx] = (uint16_t)(out_val + 0.5);
}


// ============================================================================
// Host entry point
// ============================================================================

void tbc_resample(const double* d_demod,
                  const double* d_linelocs,
                  uint16_t* d_tbc_luma,
                  int* d_oob_pixel_count,
                  int* d_bad_geom_line_count,
                  int* d_sync_like_pixel_count,
                  int* d_sync_like_line_counts,
                  bool mark_fallback_white,
                  bool mark_sync_like_white,
                  int num_fields,
                  int total_demod_samples,
                  const VideoFormat& fmt)
{
    int pixels_per_field = fmt.output_field_lines * fmt.output_line_len;
    int total_pixels = num_fields * pixels_per_field;
    int active_video_start = (fmt.system == VideoSystem::NTSC) ? 134 : 185;
    int active_video_end = (fmt.system == VideoSystem::NTSC) ? 894 : 1107;

    int threads = 256;
    int blocks = (total_pixels + threads - 1) / threads;

    k_tbc_resample<<<blocks, threads>>>(
        d_demod,
        d_linelocs,
        d_tbc_luma,
        d_oob_pixel_count,
        d_bad_geom_line_count,
        d_sync_like_pixel_count,
        d_sync_like_line_counts,
        mark_fallback_white ? 1 : 0,
        mark_sync_like_white ? 1 : 0,
        num_fields,
        fmt.lines_per_frame,
        fmt.output_field_lines,
        fmt.output_line_len,
        fmt.active_line_start,
        active_video_start,
        active_video_end,
        total_demod_samples,
        fmt.samples_per_line,
        fmt.ire0,
        fmt.hz_ire,
        fmt.vsync_ire,
        fmt.pulse_threshold_hz,
        fmt.output_zero,
        fmt.output_scale);
}
