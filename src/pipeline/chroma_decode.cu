#include "pipeline/chroma_decode.h"
#include <cstdio>

void chroma_decode(const double* d_raw,
                   const double* d_linelocs,
                   uint16_t* d_tbc_chroma,
                   int num_fields,
                   const VideoFormat& fmt) {
    // TODO: Implement VHS chroma decode
    //
    // Sub-kernels:
    //
    // A. Chroma bandpass extraction (frequency domain):
    //    - FFT the raw signal
    //    - Apply bandpass centered at chroma_under (±~500 kHz)
    //    - IFFT back to time domain
    //    This is the color-under signal.
    //
    // B. Heterodyne upconversion:
    //    - Generate carrier at het_freq = fsc + chroma_under
    //    - Multiply chroma band × carrier → shifts to fsc
    //    - Low-pass filter to remove image frequency
    //
    // C. Track phase detection (per field):
    //    - VHS alternates head azimuth; chroma phase depends on track
    //    - Try both phase offsets, measure burst amplitude
    //    - Pick the phase with stronger/cleaner burst
    //
    // D. ACC normalization (per line):
    //    - Measure burst RMS in the back porch region
    //    - Scale line by burst_abs_ref / burst_rms
    //
    // E. TBC resample (same as luma but for chroma):
    //    - Use linelocs to resample to 4*fsc output rate
    //    - Output as uint16 centered at 32768
    //
    // NOTE: Input must be in s16-equivalent range. If source is u8,
    // the pipeline normalizes during upload: (u8 - 128) * 256.
    // This was the fix for the chroma bug in the Python decoder.

    fprintf(stderr, "  [chroma_decode] STUB: %d fields\n", num_fields);
}
