#pragma once
#include "format/video_format.h"

// Kernel 4: Refine Line Locations via Hsync Correlation
//
// For each scanline, cross-correlate a window around the estimated line start
// with an ideal hsync waveform to find the sub-sample-accurate position.
//
// This replaces the Cython refine_linelocs_hsync() in the Python decoder.
//
// N_fields × lines_per_field × ~40-sample correlations = embarrassingly parallel.
void hsync_refine(const double* d_demod_05,
                  double* d_linelocs,
                  int num_fields,
                  const VideoFormat& fmt);
