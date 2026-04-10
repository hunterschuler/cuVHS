#pragma once
#include <cstdint>
#include <vector>
#include "format/video_format.h"

// Chroma state carried across batch boundaries.
// Prevents track re-detection at each batch start, which causes
// momentary greyscale dropout when the detector picks the wrong track.
struct ChromaState {
    bool valid = false;             // false = first batch, do full auto-detect
    int current_track = 0;          // track parity (0 or 1)
    double good_metric_threshold = 0.0;  // burst cancellation threshold for track flip detection
    int cycle_start = 0;            // NTSC 8-field phase cycle position
    int prev_quadrant = -1;         // burst phase quadrant from last field (for phase delta)
};

// Kernel 6: Chroma Decode (VHS Color-Under)
//
// VHS records chroma heterodyned down to ~629 kHz (NTSC) / ~626 kHz (PAL).
// This kernel extracts the color-under signal and upconverts to colorburst freq.
//
// Pipeline (all GPU except K1 source production):
//   1. Consume K1 demod_burst-equivalent color-under source
//   2. TBC resample source + heterodyne with per-line phase rotation
//   3. Per-line cuFFT: bandpass filter centered at fsc
//   4. Per-line: burst measurement + ACC normalization
//   5. Convert to uint16 centered at 32768
//
// Needs: d_chroma_source (K1 demod_burst-equivalent signal), d_linelocs (from K3/K4)
// Output: d_tbc_chroma (uint16, same geometry as d_tbc_luma)
//         field_phase_ids: per-field NTSC-family phase ID (1-4), empty for PAL
//
// state: if non-null, carries track/phase info from previous batch.
//        On first batch, pass state with valid=false for auto-detection.
//        On return, state is updated with end-of-batch values.
void chroma_decode(const double* d_chroma_source,
                   double* d_linelocs,
                   const int* d_is_first_field,
                   uint16_t* d_tbc_chroma,
                   int num_fields,
                   int total_raw_samples,
                   const VideoFormat& fmt,
                   std::vector<int>& field_phase_ids,
                   ChromaState* state = nullptr,
                   size_t raw_offset = 0);
