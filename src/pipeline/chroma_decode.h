#pragma once
#include <cstdint>
#include "format/video_format.h"

// Kernel 6: Chroma Decode (VHS Color-Under)
//
// VHS records chroma heterodyned down to ~629 kHz (NTSC) / ~626 kHz (PAL).
// This kernel extracts the color-under signal and upconverts to colorburst freq.
//
// Steps:
//   1. Bandpass filter raw signal around chroma_under frequency
//   2. Heterodyne up: multiply by cos/sin at (fsc + chroma_under) to shift to fsc
//   3. Track phase detection (try both phases, pick better burst)
//   4. Burst measurement and ACC normalization per line
//   5. Comb filtering with adjacent field (optional)
//   6. TBC resample chroma to output rate
//
// Note: needs raw signal (not demod), plus linelocs for TBC alignment.
void chroma_decode(const double* d_raw,
                   const double* d_linelocs,
                   uint16_t* d_tbc_chroma,
                   int num_fields,
                   const VideoFormat& fmt);
