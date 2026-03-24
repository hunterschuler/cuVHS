#pragma once
#include "format/video_format.h"

// Kernel 2: Find Sync Pulses
//
// From the sync-demodulated signal (demod_05), detect horizontal and vertical
// sync pulses by thresholding and edge detection.
//
// Output: per-field arrays of (start, length, level) tuples.
// d_pulses layout: [num_fields][MAX_PULSES][3] where [3] = {start, length, level}
// MAX_PULSES = 1024 (more than enough for any field)
void sync_pulses(const double* d_demod_05,
                 int* d_pulses,
                 int num_fields,
                 size_t samples_per_field,
                 const VideoFormat& fmt);
