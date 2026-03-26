#include "pipeline/fm_demod.h"
#include <cuda_runtime.h>
#include <cufft.h>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>
#include <complex>

// ============================================================
// Constants
// ============================================================
static const double PI  = 3.14159265358979323846;
static const double TAU = 2.0 * PI;

// ============================================================
// CPU helpers: FFT-friendly size (7-smooth: factors of 2,3,5,7 only)
// ============================================================
static bool is_7smooth(int n) {
    while (n % 2 == 0) n /= 2;
    while (n % 3 == 0) n /= 3;
    while (n % 5 == 0) n /= 5;
    while (n % 7 == 0) n /= 7;
    return n == 1;
}

static int next_fft_size(int n) {
    while (!is_7smooth(n)) n++;
    return n;
}

// ============================================================
// CPU helpers: Butterworth filter magnitude response
//
// Exact magnitude |H(e^jw)| for digital Butterworth filters
// designed via bilinear transform (matches scipy.signal.butter).
// Used once at init to build filter arrays on CPU, then uploaded to GPU.
// ============================================================

static double butter_lpf_mag(double omega, double omega_c, int order) {
    if (omega_c <= 0.0 || omega_c >= PI) return 1.0;
    double tc = tan(omega_c / 2.0);
    if (tc == 0.0) return 1.0;
    double t = tan(omega / 2.0);
    double ratio = t / tc;
    return 1.0 / sqrt(1.0 + pow(ratio, 2.0 * order));
}

static double butter_hpf_mag(double omega, double omega_c, int order) {
    if (omega <= 0.0) return 0.0;
    if (omega_c <= 0.0 || omega_c >= PI) return 1.0;
    double tc = tan(omega_c / 2.0);
    double t = tan(omega / 2.0);
    if (t == 0.0) return 0.0;
    double ratio = tc / t;
    return 1.0 / sqrt(1.0 + pow(ratio, 2.0 * order));
}

static double butter_bpf_mag(double omega, double omega_l, double omega_h, int order) {
    if (omega <= 0.0 || omega >= PI) return 0.0;
    double tl = tan(omega_l / 2.0);
    double th = tan(omega_h / 2.0);
    double t  = tan(omega / 2.0);
    double BW = th - tl;
    if (BW <= 0.0 || t == 0.0) return 0.0;
    double S = (t * t - tl * th) / (BW * t);
    return 1.0 / sqrt(1.0 + pow(S, 2.0 * order));
}

// ============================================================
// CPU helpers: Supergaussian LPF
// Matches Python: exp(-2 * ((2*f*(ln2/2)^(1/(2*order)))/freq)^(2*order))
// ============================================================
static double supergauss_mag(double f_hz, double corner_freq, int order) {
    double ln2_half = log(2.0) / 2.0;
    double scale = pow(ln2_half, 1.0 / (2.0 * order));
    double x = 2.0 * f_hz * scale / corner_freq;
    return exp(-2.0 * pow(x, 2.0 * order));
}

// ============================================================
// CPU helpers: Shelf filter (matching Python gen_shelf / FMDeEmphasisB)
// Based on Audio EQ Cookbook: https://www.w3.org/2011/audio/audio-eq-cookbook.html
// ============================================================
static void gen_shelf_high(double f0, double dbgain, double fs, double Q,
                           double b[3], double a[3]) {
    double A = pow(10.0, dbgain / 40.0);
    double w0 = TAU * f0 / fs;
    double alpha = sin(w0) / (2.0 * Q);
    double cosw0 = cos(w0);
    double sqA = sqrt(A);

    b[0] = A * ((A + 1) + (A - 1) * cosw0 + 2 * sqA * alpha);
    b[1] = -2 * A * ((A - 1) + (A + 1) * cosw0);
    b[2] = A * ((A + 1) + (A - 1) * cosw0 - 2 * sqA * alpha);
    a[0] = (A + 1) - (A - 1) * cosw0 + 2 * sqA * alpha;
    a[1] = 2 * ((A - 1) - (A + 1) * cosw0);
    a[2] = (A + 1) - (A - 1) * cosw0 - 2 * sqA * alpha;
}

static std::complex<double> freqz_biquad(const double num[3], const double den[3], double omega) {
    std::complex<double> z1 = std::exp(std::complex<double>(0.0, -omega));
    std::complex<double> z2 = z1 * z1;
    std::complex<double> N = num[0] + num[1] * z1 + num[2] * z2;
    std::complex<double> D = den[0] + den[1] * z1 + den[2] * z2;
    return N / D;
}

// ============================================================
// CPU helpers: FIR lowpass (matching scipy.signal.firwin, Hamming window)
// ============================================================
static std::vector<double> firwin_lpf(int num_taps, double cutoff_normalized) {
    std::vector<double> h(num_taps);
    double alpha = (num_taps - 1) / 2.0;
    double sum = 0.0;

    for (int n = 0; n < num_taps; n++) {
        double m = n - alpha;
        double sinc_val;
        if (fabs(m) < 1e-12) {
            sinc_val = cutoff_normalized;
        } else {
            sinc_val = sin(PI * cutoff_normalized * m) / (PI * m);
        }
        double window = 0.54 - 0.46 * cos(TAU * n / (num_taps - 1));
        h[n] = sinc_val * window;
        sum += h[n];
    }
    for (int n = 0; n < num_taps; n++) h[n] /= sum;
    return h;
}

static std::complex<double> fir_freqz(const std::vector<double>& h, double omega,
                                       int delay_compensate = 0) {
    std::complex<double> H(0.0, 0.0);
    for (size_t n = 0; n < h.size(); n++) {
        H += h[n] * std::exp(std::complex<double>(0.0, -(double)n * omega));
    }
    if (delay_compensate > 0) {
        H *= std::exp(std::complex<double>(0.0, omega * delay_compensate));
    }
    return H;
}

// ============================================================
// CUDA Kernels
// ============================================================

// Copy samples_per_field doubles per field into fft_size-strided buffer, zero-padding.
__global__ void k_copy_and_pad(
    const double* src,           // [num_fields x samples_per_field]
    double* dst,                 // [num_fields x fft_size]
    int samples_per_field,
    int fft_size,
    int total_dst_samples)       // num_fields * fft_size
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_dst_samples) return;
    int field = idx / fft_size;
    int pos   = idx % fft_size;
    if (pos < samples_per_field) {
        dst[idx] = src[(size_t)field * samples_per_field + pos];
    } else {
        dst[idx] = 0.0;
    }
}

// Copy first samples_per_field doubles per field from fft_size-strided buffer.
__global__ void k_trim(
    const double* src,           // [num_fields x fft_size]
    double* dst,                 // [num_fields x samples_per_field]
    int samples_per_field,
    int fft_size,
    int total_dst_samples)       // num_fields * samples_per_field
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_dst_samples) return;
    int field = idx / samples_per_field;
    int pos   = idx % samples_per_field;
    dst[idx] = src[(size_t)field * fft_size + pos];
}

// Apply RF filter (real) to R2C output (complex): fft[k] *= rf[k]
__global__ void k_apply_rf_filter(
    cufftDoubleComplex* fft_data,
    const double* rf_filter,
    int freq_bins,
    int total_bins)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_bins) return;
    int bin = idx % freq_bins;
    double rf = rf_filter[bin];
    fft_data[idx].x *= rf;
    fft_data[idx].y *= rf;
}

// Expand half-spectrum (freq_bins) to full (fft_size) for C2C inverse.
__global__ void k_expand_half_to_full(
    const cufftDoubleComplex* half,
    cufftDoubleComplex* full,
    int freq_bins,
    int fft_size,
    int num_fields)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_fields * fft_size;
    if (idx >= total) return;
    int field = idx / fft_size;
    int k = idx % fft_size;
    if (k < freq_bins) {
        full[idx] = half[(size_t)field * freq_bins + k];
    } else {
        full[idx].x = 0.0;
        full[idx].y = 0.0;
    }
}

// Parallel atan2: complex analytic signal -> phase angles
__global__ void k_compute_angles(
    const cufftDoubleComplex* analytic,
    double* angles,
    int total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    angles[idx] = atan2(analytic[idx].y, analytic[idx].x);
}

// Compute RF envelope magnitude from complex analytic signal: sqrt(re² + im²)
// Must be called BEFORE angle extraction, which overwrites the analytic buffer.
__global__ void k_compute_envelope(
    const cufftDoubleComplex* __restrict__ analytic,
    double* __restrict__ envelope,
    int spf,
    int fft_size,
    int total_fields)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = total_fields * spf;
    if (idx >= total) return;
    int field = idx / spf;
    int sample = idx % spf;
    int src = field * fft_size + sample;
    double re = analytic[src].x, im = analytic[src].y;
    envelope[idx] = sqrt(re * re + im * im);
}

// Sequential phase unwrap: one thread per field.
// Matches Python: ediff1d(angles, to_begin=0) -> unwrap -> clamp [0,2pi] -> scale
__global__ void k_unwrap_to_hz(
    const double* angles,
    double* demod,
    int fft_size,
    int samples_per_field,
    double freq_hz,
    int num_fields)
{
    int field = blockIdx.x * blockDim.x + threadIdx.x;
    if (field >= num_fields) return;

    const double* a = angles + (size_t)field * fft_size;
    double* out = demod + (size_t)field * samples_per_field;
    int N = (fft_size < samples_per_field) ? fft_size : samples_per_field;

    double scale = freq_hz / TAU;
    double prev_raw_dangle = 0.0;
    double correction = 0.0;

    out[0] = 0.0;

    for (int i = 1; i < N; i++) {
        double dangle = a[i] - a[i - 1];
        double delta = dangle - prev_raw_dangle;
        prev_raw_dangle = dangle;

        if (delta > PI)       correction -= TAU;
        else if (delta < -PI) correction += TAU;

        double unwrapped = dangle + correction;
        double clamped = fmod(unwrapped, TAU);
        if (clamped < 0.0) clamped += TAU;

        out[i] = clamped * scale;
    }
}

// Apply complex filter: out[k] = in[k] * filter[k]
__global__ void k_apply_complex_filter(
    const cufftDoubleComplex* in,
    cufftDoubleComplex* out,
    const cufftDoubleComplex* filter,
    int freq_bins,
    int total_bins)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_bins) return;
    int bin = idx % freq_bins;
    cufftDoubleComplex f = filter[bin];
    cufftDoubleComplex v = in[idx];
    out[idx].x = v.x * f.x - v.y * f.y;
    out[idx].y = v.x * f.y + v.y * f.x;
}

// ============================================================
// Scratch memory budget (for pipeline VRAM planning)
// ============================================================
size_t FMDemodState::scratch_bytes_per_field(int samples_per_field) {
    int fs = next_fft_size(samples_per_field);
    int fb = fs / 2 + 1;
    return (size_t)fb * sizeof(cufftDoubleComplex)     // d_fft_half
         + (size_t)fs * sizeof(cufftDoubleComplex)     // d_analytic (multi-purpose scratch)
         + (size_t)fb * sizeof(cufftDoubleComplex);    // d_post_fft
}

// ============================================================
// Init
// ============================================================
bool FMDemodState::init(const VideoFormat& fmt, int max_batch_size, int samples_per_field_override) {
    max_batch = max_batch_size;
    int spf = (samples_per_field_override > 0) ? samples_per_field_override : fmt.samples_per_field;
    fft_size  = next_fft_size(spf);
    freq_bins = fft_size / 2 + 1;

    fprintf(stderr, "  [FM demod] FFT size: %d (field: %d, pad: +%d)  freq bins: %d  batch: %d\n",
            fft_size, spf, fft_size - spf, freq_bins, max_batch);

    double fs      = fmt.sample_rate;
    double nyquist = fs / 2.0;

    // ---- VHS NTSC SP RF filter parameters ----
    double bpf_low        = 500000.0;
    double bpf_high       = 6500000.0;
    int    bpf_order      = 8;
    double lpf_extra_freq = 6000000.0;
    int    lpf_extra_ord  = 25;
    double hpf_extra_freq = 1200000.0;
    int    hpf_extra_ord  = 20;

    double w_bpf_lo = PI * bpf_low / nyquist;
    double w_bpf_hi = PI * bpf_high / nyquist;
    double w_lpf    = PI * lpf_extra_freq / nyquist;
    double w_hpf    = PI * hpf_extra_freq / nyquist;

    // ---- RF filter x Hilbert (freq_bins reals) ----
    std::vector<double> h_rf(freq_bins);
    for (int k = 0; k < freq_bins; k++) {
        double omega = PI * (double)k / (double)(freq_bins - 1);
        double bpf = butter_bpf_mag(omega, w_bpf_lo, w_bpf_hi, bpf_order);
        double lpf = butter_lpf_mag(omega, w_lpf, lpf_extra_ord);
        double hpf = butter_hpf_mag(omega, w_hpf, hpf_extra_ord);
        double hil = (k == 0 || k == freq_bins - 1) ? 1.0 : 2.0;
        h_rf[k] = bpf * lpf * hpf * hil;
    }

    // ---- Deemphasis: inverse of high-shelf ----
    double deemph_mid  = 273755.82;
    double deemph_gain = 13.9794;
    double deemph_q    = 0.462088186;
    double b_shelf[3], a_shelf[3];
    gen_shelf_high(deemph_mid, deemph_gain, fs, deemph_q, b_shelf, a_shelf);

    // ---- Video LPF: supergaussian 6.6 MHz order 9 ----
    double video_lpf_freq  = 6600000.0;
    int    video_lpf_order = 9;

    // ---- Sync FIR: 65-tap 0.5 MHz LPF ----
    double sync_cutoff_norm = 500000.0 / nyquist;
    auto   sync_fir = firwin_lpf(65, sync_cutoff_norm);
    f05_offset = 32;

    // ---- Build FVideo and FVideo05 (pre-scaled by 1/N for cuFFT normalization) ----
    std::vector<cufftDoubleComplex> h_fv(freq_bins), h_fv05(freq_bins);
    double inv_n = 1.0 / (double)fft_size;

    for (int k = 0; k < freq_bins; k++) {
        double omega = PI * (double)k / (double)(freq_bins - 1);
        double f_hz  = (double)k * nyquist / (double)(freq_bins - 1);

        std::complex<double> deemp = freqz_biquad(a_shelf, b_shelf, omega);
        double vlpf = supergauss_mag(f_hz, video_lpf_freq, video_lpf_order);
        std::complex<double> sync = fir_freqz(sync_fir, omega, f05_offset);

        std::complex<double> fv = deemp * vlpf * inv_n;
        h_fv[k].x = fv.real();
        h_fv[k].y = fv.imag();

        std::complex<double> fv05 = deemp * vlpf * sync * inv_n;
        h_fv05[k].x = fv05.real();
        h_fv05[k].y = fv05.imag();
    }

    // ---- Upload filters to GPU ----
    cudaMalloc(&d_rf_filter, freq_bins * sizeof(double));
    cudaMemcpy(d_rf_filter, h_rf.data(), freq_bins * sizeof(double), cudaMemcpyHostToDevice);

    cudaMalloc(&d_fvideo, freq_bins * sizeof(cufftDoubleComplex));
    cudaMemcpy(d_fvideo, h_fv.data(), freq_bins * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice);

    cudaMalloc(&d_fvideo05, freq_bins * sizeof(cufftDoubleComplex));
    cudaMemcpy(d_fvideo05, h_fv05.data(), freq_bins * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice);

    // ---- Allocate scratch buffers ----
    cudaMalloc(&d_fft_half, (size_t)max_batch * freq_bins * sizeof(cufftDoubleComplex));
    cudaMalloc(&d_analytic, (size_t)max_batch * fft_size  * sizeof(cufftDoubleComplex));
    cudaMalloc(&d_post_fft, (size_t)max_batch * freq_bins * sizeof(cufftDoubleComplex));

    // ---- Create cuFFT plans ----
    {
        int n[] = { fft_size };
        cufftResult r = cufftPlanMany(&plan_r2c, 1, n,
                                       NULL, 1, fft_size,
                                       NULL, 1, freq_bins,
                                       CUFFT_D2Z, max_batch);
        if (r != CUFFT_SUCCESS) {
            fprintf(stderr, "cuFFT R2C plan failed: %d\n", r);
            return false;
        }
    }
    {
        int n[] = { fft_size };
        cufftResult r = cufftPlanMany(&plan_c2c_inv, 1, n,
                                       NULL, 1, fft_size,
                                       NULL, 1, fft_size,
                                       CUFFT_Z2Z, max_batch);
        if (r != CUFFT_SUCCESS) {
            fprintf(stderr, "cuFFT C2C plan failed: %d\n", r);
            return false;
        }
    }
    {
        int n[] = { fft_size };
        cufftResult r = cufftPlanMany(&plan_c2r, 1, n,
                                       NULL, 1, freq_bins,
                                       NULL, 1, fft_size,
                                       CUFFT_Z2D, max_batch);
        if (r != CUFFT_SUCCESS) {
            fprintf(stderr, "cuFFT C2R plan failed: %d\n", r);
            return false;
        }
    }

    fprintf(stderr, "  [FM demod] Filters and plans ready.\n");
    return true;
}

void FMDemodState::destroy() {
    if (plan_r2c)     { cufftDestroy(plan_r2c);     plan_r2c = 0; }
    if (plan_c2c_inv) { cufftDestroy(plan_c2c_inv); plan_c2c_inv = 0; }
    if (plan_c2r)     { cufftDestroy(plan_c2r);     plan_c2r = 0; }

    if (d_rf_filter) { cudaFree(d_rf_filter); d_rf_filter = nullptr; }
    if (d_fvideo)    { cudaFree(d_fvideo);    d_fvideo    = nullptr; }
    if (d_fvideo05)  { cudaFree(d_fvideo05);  d_fvideo05  = nullptr; }
    if (d_fft_half)  { cudaFree(d_fft_half);  d_fft_half  = nullptr; }
    if (d_analytic)  { cudaFree(d_analytic);  d_analytic  = nullptr; }
    if (d_post_fft)  { cudaFree(d_post_fft);  d_post_fft  = nullptr; }
}

bool fm_demod_init(FMDemodState& state, const VideoFormat& fmt, int max_batch) {
    return state.init(fmt, max_batch);
}

// ============================================================
// FM Demodulation: run the full pipeline on a batch of fields
//
// All GPU. d_analytic is reused as multi-purpose scratch:
//   - padded input buffer (as double*)
//   - full-spectrum for C2C inverse (as cufftDoubleComplex*)
//   - angles buffer (as double*, after C2C)
//   - C2R output buffer (as double*, after phase unwrap)
// ============================================================
void fm_demod(FMDemodState& state,
              const double* d_raw,
              double* d_demod,
              double* d_demod_05,
              double* d_envelope,
              int num_fields,
              size_t samples_per_field,
              const VideoFormat& fmt)
{
    int fb  = state.freq_bins;
    int fs  = state.fft_size;
    int spf = (int)samples_per_field;
    int T   = 256;   // threads per block

    // d_analytic memory is fft_size * max_batch * 16 bytes.
    // As double*, that's fft_size * max_batch * 2 doubles.
    // We reuse it for several purposes below.
    double* d_scratch = reinterpret_cast<double*>(state.d_analytic);

    // ========== PART A: RF filtering + analytic signal ==========

    // Step 1: Copy raw input into fft_size-strided padded buffer
    // d_scratch layout: [field0: fft_size doubles, field1: fft_size doubles, ...]
    // (only uses first fft_size doubles per field out of 2*fft_size available)
    {
        int total = num_fields * fs;
        int blocks = (total + T - 1) / T;
        k_copy_and_pad<<<blocks, T>>>(d_raw, d_scratch, spf, fs, total);
    }

    // Step 2: Forward FFT (R2C) of padded raw input
    cufftExecD2Z(state.plan_r2c,
                 reinterpret_cast<cufftDoubleReal*>(d_scratch),
                 state.d_fft_half);

    // Step 3: Apply RF filter x Hilbert weight
    {
        int total = num_fields * fb;
        int blocks = (total + T - 1) / T;
        k_apply_rf_filter<<<blocks, T>>>(state.d_fft_half, state.d_rf_filter, fb, total);
    }

    // Step 4: Expand half-spectrum to full (negative freqs = 0)
    // Now d_analytic is used as cufftDoubleComplex* (its native type)
    {
        int total = num_fields * fs;
        int blocks = (total + T - 1) / T;
        k_expand_half_to_full<<<blocks, T>>>(
            state.d_fft_half, state.d_analytic, fb, fs, num_fields);
    }

    // Step 5: C2C inverse FFT -> complex analytic signal (in-place)
    cufftExecZ2Z(state.plan_c2c_inv, state.d_analytic, state.d_analytic, CUFFT_INVERSE);

    // Step 5b: Compute RF envelope from analytic signal (before angle extraction overwrites it)
    if (d_envelope) {
        int total = num_fields * spf;
        int blocks = (total + T - 1) / T;
        k_compute_envelope<<<blocks, T>>>(state.d_analytic, d_envelope, spf, fs, num_fields);
    }

    // Step 6a: Extract phase angles in parallel
    // Reuse first half of d_analytic memory as double* for angles.
    // k_compute_angles reads complex at [16i..16i+15] and writes double at [8i..8i+7].
    // No aliasing conflict: output bytes never overlap input bytes for the same element.
    double* d_angles = reinterpret_cast<double*>(state.d_analytic);
    {
        int total = num_fields * fs;
        int blocks = (total + T - 1) / T;
        k_compute_angles<<<blocks, T>>>(state.d_analytic, d_angles, total);
    }

    // Step 6b: Sequential phase unwrap -> instantaneous frequency in Hz
    // Output goes to d_demod (caller's buffer, samples_per_field per field)
    {
        int blocks = (num_fields + T - 1) / T;
        k_unwrap_to_hz<<<blocks, T>>>(
            d_angles, d_demod, fs, spf, fmt.sample_rate, num_fields);
    }

    // ========== PART B: post-demod filtering ==========

    // Step 7: Copy demod into padded buffer for R2C
    // d_scratch = reinterpret_cast<double*>(d_analytic)  — already set
    {
        int total = num_fields * fs;
        int blocks = (total + T - 1) / T;
        k_copy_and_pad<<<blocks, T>>>(d_demod, d_scratch, spf, fs, total);
    }

    // Step 8: Forward FFT (R2C) of padded demod signal
    cufftExecD2Z(state.plan_r2c,
                 reinterpret_cast<cufftDoubleReal*>(d_scratch),
                 state.d_fft_half);

    // Step 9: Apply FVideo, C2R -> video output
    {
        int total = num_fields * fb;
        int blocks = (total + T - 1) / T;
        k_apply_complex_filter<<<blocks, T>>>(
            state.d_fft_half, state.d_post_fft, state.d_fvideo, fb, total);
    }
    // C2R output goes to d_scratch (fft_size-strided), then we trim to d_demod
    cufftExecZ2D(state.plan_c2r, state.d_post_fft,
                 reinterpret_cast<cufftDoubleReal*>(d_scratch));
    {
        int total = num_fields * spf;
        int blocks = (total + T - 1) / T;
        k_trim<<<blocks, T>>>(d_scratch, d_demod, spf, fs, total);
    }

    // Step 10: Apply FVideo05, C2R -> sync output
    // d_fft_half is still intact (C2R only consumed d_post_fft + wrote d_scratch)
    {
        int total = num_fields * fb;
        int blocks = (total + T - 1) / T;
        k_apply_complex_filter<<<blocks, T>>>(
            state.d_fft_half, state.d_post_fft, state.d_fvideo05, fb, total);
    }
    cufftExecZ2D(state.plan_c2r, state.d_post_fft,
                 reinterpret_cast<cufftDoubleReal*>(d_scratch));
    {
        int total = num_fields * spf;
        int blocks = (total + T - 1) / T;
        k_trim<<<blocks, T>>>(d_scratch, d_demod_05, spf, fs, total);
    }
    // Delay compensation for sync FIR is pre-baked into d_fvideo05 filter.
}
