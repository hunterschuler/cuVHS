#include "pipeline/tbc_resample.h"
#include <cstdio>

void tbc_resample(const double* d_demod,
                  const double* d_linelocs,
                  uint16_t* d_tbc_luma,
                  int num_fields,
                  const VideoFormat& fmt) {
    // TODO: Implement TBC resampling
    //
    // For each output pixel (field, line, sample):
    //   1. Compute input position:
    //      input_pos = linelocs[field][line]
    //                + sample * (linelocs[field][line+1] - linelocs[field][line])
    //                  / output_line_len
    //   2. Cubic interpolation from d_demod at input_pos
    //   3. Apply level scaling (IRE to uint16 range)
    //   4. Clamp to [0, 65535] and write to d_tbc_luma
    //
    // GPU strategy:
    //   One thread per output sample.
    //   Grid: (output_line_len, lines_per_field, num_fields)
    //   Memory-bound — optimize for coalesced reads from d_demod.

    fprintf(stderr, "  [tbc_resample] STUB: %d fields\n", num_fields);
}
