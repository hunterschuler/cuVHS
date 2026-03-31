#pragma once
#include "format/video_format.h"

// Kernel 4: Refine Line Locations via Hsync Zero-Crossing
//
// Two-pass zero-crossing detection on demod_05 (sync signal in Hz domain):
//
//   Pass 1: Find initial zero-crossing at pulse threshold (-20 IRE)
//           within ±1 µs of the estimated lineloc.
//
//   Pass 2: Measure actual sync and porch levels from the signal,
//           then find refined crossing at (sync + porch) / 2 midpoint.
//           Linear interpolation between samples for sub-sample accuracy.
//
// Matches Python refine_linelocs_hsync() from vhsdecode/sync.pyx.
//
// One CUDA thread per line (num_fields × lines_per_frame).
void hsync_refine(const double* d_demod_05,
                  double* d_linelocs,
                  int num_fields,
                  int total_demod_samples,
                  const VideoFormat& fmt);

void hsync_refine_debug_analyze(const double* d_demod_05,
                                const double* d_linelocs_before,
                                const double* d_linelocs_after,
                                int* d_large_delta_count,
                                int* d_isolated_jump_count,
                                int* d_refined_sync_like_count,
                                int num_fields,
                                int total_demod_samples,
                                const VideoFormat& fmt);
