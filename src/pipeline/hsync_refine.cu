#include "pipeline/hsync_refine.h"
#include <cstdio>

void hsync_refine(const double* d_demod_05,
                  double* d_linelocs,
                  int num_fields,
                  const VideoFormat& fmt) {
    // TODO: Implement hsync correlation refinement
    //
    // For each line in each field:
    //   1. Load ~40 samples around linelocs[field][line] from demod_05
    //   2. Cross-correlate with ideal hsync template (precomputed, in shared mem)
    //   3. Find correlation peak (sub-sample via parabolic interpolation)
    //   4. Update linelocs[field][line] with refined position
    //
    // GPU strategy:
    //   One thread per line (num_fields × lines_per_field threads total).
    //   Hsync template loaded into shared memory once per block.

    fprintf(stderr, "  [hsync_refine] STUB: %d fields\n", num_fields);
}
