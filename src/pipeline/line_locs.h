#pragma once
#include "format/video_format.h"

// Kernel 3: Classify Pulses + Compute Line Locations
//
// From pulse arrays, determine field boundaries, compute mean line length,
// and assign sample positions for each scanline.
//
// d_linelocs output: [num_fields][lines_per_frame] — sample position of each line start
void line_locs(const int* d_pulses,
               double* d_linelocs,
               int num_fields,
               const VideoFormat& fmt);
