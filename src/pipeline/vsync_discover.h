#pragma once
#include "format/video_format.h"

// Scans the demodulated sync stream for pulses matching VSYNC serration widths.
// Populates an array of indices where these pulses begin.
void discover_vsyncs(const double* d_demod_05,
                     int* d_candidate_indices,
                     int* d_candidate_count,
                     int candidate_capacity,
                     int total_samples,
                     const VideoFormat& fmt);
