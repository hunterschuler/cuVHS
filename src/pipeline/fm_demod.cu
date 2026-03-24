#include "pipeline/fm_demod.h"
#include <cstdio>

void fm_demod(const double* d_raw,
              double* d_demod,
              double* d_demod_05,
              int num_fields,
              size_t samples_per_field,
              const VideoFormat& fmt) {
    // TODO: Implement batched FM demodulation
    //
    // Steps:
    // 1. Batched cuFFT R2C (num_fields plans, samples_per_field each)
    // 2. Apply frequency-domain filters:
    //    - Video bandpass (luma carrier ± bandwidth)
    //    - Deemphasis curve
    //    - Chroma trap (notch at chroma_under)
    //    - Separate low-pass for sync signal (demod_05)
    // 3. Batched cuFFT C2R inverse
    // 4. Hilbert transform + angle unwrap → instantaneous frequency
    //    - This gives the demodulated baseband video
    //
    // Filter coefficients are derived from fmt (sample_rate, luma_carrier, etc.)
    // and should be precomputed once at pipeline init, not per-batch.

    fprintf(stderr, "  [fm_demod] STUB: %d fields × %zu samples\n",
            num_fields, samples_per_field);
}
