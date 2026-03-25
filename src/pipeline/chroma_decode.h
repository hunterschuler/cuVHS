#pragma once
#include <cstdint>
#include <vector>
#include "format/video_format.h"

// Kernel 6: Chroma Decode (VHS Color-Under)
//
// VHS records chroma heterodyned down to ~629 kHz (NTSC) / ~626 kHz (PAL).
// This kernel extracts the color-under signal and upconverts to colorburst freq.
//
// Pipeline (all GPU):
//   1. Pre-bandpass filter d_raw at capture rate (60 kHz – 1.2 MHz)
//   2. TBC resample filtered signal + heterodyne with per-line phase rotation
//   3. Per-line cuFFT: bandpass filter centered at fsc
//   4. Per-line: burst measurement + ACC normalization
//   5. Convert to uint16 centered at 32768
//
// Needs: d_raw (raw RF signal), d_linelocs (from K3/K4)
// d_scratch: writable buffer same size as d_raw (num_fields × samples_per_field doubles).
//            Used for bandpass-filtered raw data. Can reuse d_demod since K5 is done.
// Output: d_tbc_chroma (uint16, same geometry as d_tbc_luma)
//         field_phase_ids: per-field NTSC phase ID (1-4), empty for PAL
void chroma_decode(const double* d_raw,
                   const double* d_linelocs,
                   double* d_scratch,
                   uint16_t* d_tbc_chroma,
                   int num_fields,
                   int total_raw_samples,
                   const VideoFormat& fmt,
                   std::vector<int>& field_phase_ids);
