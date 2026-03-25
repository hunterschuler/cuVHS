#pragma once
#include <cstdint>
#include "format/video_format.h"

// Kernel 5: TBC Resample
//
// Resample demodulated video from capture rate to fixed 4*fsc output rate
// using refined line locations. Cubic interpolation.
//
// For each output sample (output_line_len × output_field_lines per field):
//   1. Map output position to input position via linelocs
//   2. Cubic interpolation from demod signal
//   3. Scale and clamp to uint16
//
// N_fields × output_line_len × output_field_lines = ~160M samples for batch of 64.
void tbc_resample(const double* d_demod,
                  const double* d_linelocs,
                  uint16_t* d_tbc_luma,
                  int num_fields,
                  int total_demod_samples,
                  const VideoFormat& fmt);
