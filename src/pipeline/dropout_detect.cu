#include "pipeline/dropout_detect.h"
#include <cstdio>

void dropout_detect(const double* d_demod,
                    uint16_t* d_tbc_luma,
                    int num_fields,
                    size_t samples_per_field,
                    const VideoFormat& fmt) {
    // TODO: Implement dropout detection and concealment
    //
    // Detection (per field):
    //   1. Compute RF envelope: abs(hilbert(demod)) or use pre-computed envelope
    //   2. Compute field-average envelope level
    //   3. Threshold: mark samples where envelope < threshold_p * field_avg
    //   4. Hysteresis: dropout starts when below threshold, ends when above
    //      threshold * hysteresis_factor
    //   5. Merge nearby dropouts (within merge_threshold samples)
    //   6. Filter out very short dropouts (< min_length samples)
    //   7. Map dropout regions from RF sample positions to TBC line/column positions
    //
    // Concealment (per dropout region in TBC):
    //   - Replace affected pixels with corresponding pixels from adjacent line
    //   - For multi-line dropouts, use the nearest clean line
    //
    // GPU strategy:
    //   Detection: one thread per sample for thresholding, then stream compaction
    //   Concealment: one thread per pixel in dropout regions

    fprintf(stderr, "  [dropout_detect] STUB: %d fields\n", num_fields);
}
