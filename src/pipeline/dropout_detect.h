#pragma once
#include <cstdint>
#include "format/video_format.h"

#define MAX_DROPOUTS_PER_FIELD 512

// Kernel 7: Dropout Detection
//
// Detects RF signal dropouts from envelope amplitude and maps them to
// TBC line/column positions for ld-tools compatible metadata output.
//
// Algorithm (matches Python vhs-decode doc.py):
//   1. Compute RF envelope: mean(raw²) per field → field-average amplitude
//   2. Threshold = field_average * dod_threshold_p (default 18%)
//   3. Scan raw signal in blocks: local RMS vs threshold with hysteresis
//   4. Merge nearby dropouts (< 30 RF samples apart)
//   5. Discard very short dropouts (< 10 RF samples)
//   6. Map RF positions to TBC line/column via linelocs
//
// Output: per-field arrays of (line, startx, endx) dropout entries,
// downloaded to host and passed to TBCWriter for JSON metadata.
void dropout_detect(const double* d_envelope,
                    const double* d_linelocs,
                    uint16_t* d_tbc_luma,
                    uint16_t* d_tbc_chroma,
                    int* d_do_lines,     // [num_fields × MAX_DROPOUTS_PER_FIELD]
                    int* d_do_starts,    // [num_fields × MAX_DROPOUTS_PER_FIELD]
                    int* d_do_ends,      // [num_fields × MAX_DROPOUTS_PER_FIELD]
                    int* d_do_count,     // [num_fields]
                    int num_fields,
                    size_t samples_per_field,
                    const VideoFormat& fmt);
