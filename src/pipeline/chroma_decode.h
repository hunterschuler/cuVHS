#pragma once
#include <cstdint>
#include "format/video_format.h"

// Kernel 6: Chroma Decode (VHS Color-Under)
//
// VHS records chroma heterodyned down to ~629 kHz (NTSC) / ~626 kHz (PAL).
// This kernel extracts the color-under signal and upconverts to colorburst freq.
//
// Pipeline (all GPU):
//   1. TBC resample d_raw using linelocs (cubic interpolation to 4*fsc)
//   2. Per-sample: multiply by heterodyne carrier at (fsc + chroma_under)
//   3. Per-line cuFFT: bandpass filter centered at fsc
//   4. Per-line: burst measurement + ACC normalization
//   5. Convert to uint16 centered at 32768
//
// Needs: d_raw (raw RF signal), d_linelocs (from K3/K4)
// Output: d_tbc_chroma (uint16, same geometry as d_tbc_luma)
void chroma_decode(const double* d_raw,
                   const double* d_linelocs,
                   uint16_t* d_tbc_chroma,
                   int num_fields,
                   int total_raw_samples,
                   const VideoFormat& fmt);
