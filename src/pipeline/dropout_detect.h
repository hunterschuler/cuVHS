#pragma once
#include <cstdint>
#include "format/video_format.h"

// Kernel 7: Dropout Detection
//
// Detect signal dropouts from the RF envelope and mark them in metadata.
// Optionally apply inline concealment (replace from adjacent line).
//
// Detection: threshold on demod envelope amplitude.
// Concealment: copy from line above or below in the TBC.
void dropout_detect(const double* d_demod,
                    uint16_t* d_tbc_luma,
                    int num_fields,
                    size_t samples_per_field,
                    const VideoFormat& fmt);
