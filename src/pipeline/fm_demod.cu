#include "pipeline/fm_demod.h"
#include <cuda_runtime.h>
#include <cufft.h>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <vector>
#include <complex>

// ============================================================
// Constants
// ============================================================
static const double PI  = 3.14159265358979323846;
static const double TAU = 2.0 * PI;

// THE FIX: The Overlap-Save Trash Zone (Absorbs FFT Ringing)
static const int OVERLAP_SAMPLES = FM_DEMOD_OVERLAP_SAMPLES;

// ============================================================
// CPU helpers
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

struct DigitalZpk {
    std::vector<std::complex<double>> z;
    std::vector<std::complex<double>> p;
    std::complex<double> k{1.0, 0.0};
};

struct IirFilter {
    std::vector<double> b;
    std::vector<double> a;
};

static std::vector<std::complex<double>> zpk_freqz_cpp(const DigitalZpk& filt,
                                                       std::size_t wor_n,
                                                       bool whole);

static std::vector<double> interp_real_cpp(const std::vector<double>& src, std::size_t dst_size) {
    std::vector<double> dst(dst_size, 0.0);
    if (src.empty() || dst_size == 0) return dst;
    if (src.size() == 1U) {
        std::fill(dst.begin(), dst.end(), src[0]);
        return dst;
    }
    const double src_scale = static_cast<double>(src.size() - 1U);
    const double dst_scale = static_cast<double>(dst_size - 1U);
    for (std::size_t i = 0; i < dst_size; ++i) {
        const double pos = (dst_scale > 0.0) ? (static_cast<double>(i) * src_scale / dst_scale) : 0.0;
        const std::size_t lo = static_cast<std::size_t>(pos);
        const std::size_t hi = std::min(lo + 1U, src.size() - 1U);
        const double frac = pos - static_cast<double>(lo);
        dst[i] = src[lo] * (1.0 - frac) + src[hi] * frac;
    }
    return dst;
}

static std::vector<std::complex<double>> interp_complex_cpp(
    const std::vector<std::complex<double>>& src,
    std::size_t dst_size)
{
    std::vector<std::complex<double>> dst(dst_size, {0.0, 0.0});
    if (src.empty() || dst_size == 0) return dst;
    if (src.size() == 1U) {
        std::fill(dst.begin(), dst.end(), src[0]);
        return dst;
    }
    const double src_scale = static_cast<double>(src.size() - 1U);
    const double dst_scale = static_cast<double>(dst_size - 1U);
    for (std::size_t i = 0; i < dst_size; ++i) {
        const double pos = (dst_scale > 0.0) ? (static_cast<double>(i) * src_scale / dst_scale) : 0.0;
        const std::size_t lo = static_cast<std::size_t>(pos);
        const std::size_t hi = std::min(lo + 1U, src.size() - 1U);
        const double frac = pos - static_cast<double>(lo);
        dst[i] = src[lo] * (1.0 - frac) + src[hi] * frac;
    }
    return dst;
}

static void first_order_lowpass_coeffs(double fs, double cutoff_hz,
                                       double& b0, double& b1, double& a1, double& zi) {
    const double k = tan(PI * cutoff_hz / fs);
    const double norm = 1.0 / (1.0 + k);
    b0 = k * norm;
    b1 = k * norm;
    a1 = (k - 1.0) * norm;
    zi = (b1 - a1 * b0) / (1.0 + a1);
}

static double supergauss_mag(double f_hz, double corner_freq, int order) {
    double ln2_half = log(2.0) / 2.0;
    double scale = pow(ln2_half, 1.0 / (2.0 * order));
    double x = 2.0 * f_hz * scale / corner_freq;
    return exp(-2.0 * pow(x, 2.0 * order));
}

static double ramp_filter_mag(double f_hz, double start_freq_hz, double boost_start,
                              double boost_max, double nyquist_hz, double max_freq_hz = 20e6) {
    if (f_hz <= start_freq_hz) return boost_start;
    if (nyquist_hz <= start_freq_hz) return boost_start;
    const double ramp_end = boost_max * (nyquist_hz / max_freq_hz);
    const double t = (f_hz - start_freq_hz) / (nyquist_hz - start_freq_hz);
    const double clamped = (t < 0.0) ? 0.0 : (t > 1.0 ? 1.0 : t);
    return boost_start + (ramp_end - boost_start) * clamped;
}

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

static std::vector<std::complex<double>> freqz_ba_cpp(const IirFilter& filter,
                                                      std::size_t wor_n,
                                                      bool whole) {
    std::vector<std::complex<double>> out(wor_n);
    const double step = whole ? (TAU / static_cast<double>(wor_n))
                              : ((wor_n > 1U) ? (PI / static_cast<double>(wor_n - 1U)) : 0.0);
    for (std::size_t i = 0; i < wor_n; ++i) {
        const double w = step * static_cast<double>(i);
        const std::complex<double> z_inv = std::exp(std::complex<double>(0.0, -w));
        std::complex<double> num{0.0, 0.0};
        std::complex<double> den{0.0, 0.0};
        std::complex<double> z_pow{1.0, 0.0};
        for (double b : filter.b) {
            num += b * z_pow;
            z_pow *= z_inv;
        }
        z_pow = {1.0, 0.0};
        for (double a : filter.a) {
            den += a * z_pow;
            z_pow *= z_inv;
        }
        out[i] = num / den;
    }
    return out;
}

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

static IirFilter make_fm_deemphasis_b_cpp(double fs, double dbgain, double mid_point, double q) {
    double b_shelf[3], a_shelf[3];
    gen_shelf_high(mid_point, dbgain, fs, q, b_shelf, a_shelf);
    IirFilter filt;
    filt.b = {a_shelf[0], a_shelf[1], a_shelf[2]};
    filt.a = {b_shelf[0], b_shelf[1], b_shelf[2]};
    return filt;
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

static DigitalZpk buttap_zpk_cpp(int order) {
    DigitalZpk out;
    for (int m = 0; m < order; ++m) {
        const double theta = PI * (2.0 * static_cast<double>(m) + 1.0 + static_cast<double>(order)) /
                             (2.0 * static_cast<double>(order));
        out.p.emplace_back(std::polar(1.0, theta));
    }
    return out;
}

static DigitalZpk lp2lp_zpk_cpp(const DigitalZpk& in, double wo) {
    DigitalZpk out;
    out.z.reserve(in.z.size());
    out.p.reserve(in.p.size());
    for (const auto& z : in.z) out.z.push_back(z * wo);
    for (const auto& p : in.p) out.p.push_back(p * wo);
    const int degree = static_cast<int>(in.p.size()) - static_cast<int>(in.z.size());
    out.k = in.k * std::pow(wo, degree);
    return out;
}

static DigitalZpk lp2hp_zpk_cpp(const DigitalZpk& in, double wo) {
    DigitalZpk out;
    out.z.reserve(in.z.size());
    out.p.reserve(in.p.size());
    for (const auto& z : in.z) out.z.push_back(wo / z);
    for (const auto& p : in.p) out.p.push_back(wo / p);
    const int degree = static_cast<int>(in.p.size()) - static_cast<int>(in.z.size());
    for (int i = 0; i < degree; ++i) out.z.emplace_back(0.0, 0.0);
    std::complex<double> num = in.k;
    for (const auto& z : in.z) num *= -z;
    std::complex<double> den{1.0, 0.0};
    for (const auto& p : in.p) den *= -p;
    out.k = num / den;
    return out;
}

static DigitalZpk lp2bp_zpk_cpp(const DigitalZpk& in, double wo, double bw) {
    DigitalZpk out;
    const int degree = static_cast<int>(in.p.size()) - static_cast<int>(in.z.size());
    out.z.reserve(in.z.size() * 2U + static_cast<std::size_t>(degree));
    out.p.reserve(in.p.size() * 2U);
    const double bw2 = bw / 2.0;
    for (const auto& z : in.z) {
        const auto temp = z * bw2;
        const auto rad = std::sqrt(temp * temp - std::complex<double>(wo * wo, 0.0));
        out.z.push_back(temp + rad);
        out.z.push_back(temp - rad);
    }
    for (const auto& p : in.p) {
        const auto temp = p * bw2;
        const auto rad = std::sqrt(temp * temp - std::complex<double>(wo * wo, 0.0));
        out.p.push_back(temp + rad);
        out.p.push_back(temp - rad);
    }
    for (int i = 0; i < degree; ++i) out.z.emplace_back(0.0, 0.0);
    out.k = in.k * std::pow(bw, degree);
    return out;
}

static DigitalZpk bilinear_zpk_cpp(const DigitalZpk& in, double fs) {
    DigitalZpk out;
    out.z.reserve(in.z.size());
    out.p.reserve(in.p.size());
    const std::complex<double> fs2{2.0 * fs, 0.0};
    for (const auto& z : in.z) out.z.push_back((fs2 + z) / (fs2 - z));
    for (const auto& p : in.p) out.p.push_back((fs2 + p) / (fs2 - p));
    const int degree = static_cast<int>(in.p.size()) - static_cast<int>(in.z.size());
    for (int i = 0; i < degree; ++i) out.z.emplace_back(-1.0, 0.0);
    std::complex<double> num = in.k;
    for (const auto& z : in.z) num *= (fs2 - z);
    std::complex<double> den{1.0, 0.0};
    for (const auto& p : in.p) den *= (fs2 - p);
    out.k = num / den;
    return out;
}

static DigitalZpk ba_lp2lp_bilinear_cpp(const std::vector<double>& b,
                                        const std::vector<double>& a,
                                        double wn_norm,
                                        bool analog) {
    auto quad_roots = [](const std::vector<double>& c) {
        std::vector<std::complex<double>> roots;
        if (c.size() == 3U) {
            const std::complex<double> disc(c[1] * c[1] - 4.0 * c[0] * c[2], 0.0);
            const auto sdisc = std::sqrt(disc);
            roots.push_back((-c[1] + sdisc) / (2.0 * c[0]));
            roots.push_back((-c[1] - sdisc) / (2.0 * c[0]));
        } else if (c.size() == 2U) {
            roots.push_back(std::complex<double>(-c[1] / c[0], 0.0));
        }
        return roots;
    };

    DigitalZpk zpk;
    zpk.z = quad_roots(b);
    zpk.p = quad_roots(a);
    zpk.k = std::complex<double>(b.empty() ? 1.0 : b[0], 0.0) /
            std::complex<double>(a.empty() ? 1.0 : a[0], 0.0);

    double warped = wn_norm;
    if (!analog) {
        const double fs = 2.0;
        warped = 2.0 * fs * std::tan(PI * wn_norm / fs);
    }
    return analog ? lp2lp_zpk_cpp(zpk, warped) : bilinear_zpk_cpp(lp2lp_zpk_cpp(zpk, warped), 2.0);
}

static std::vector<std::complex<double>> peaking_freq_response_cpp(double wn_norm,
                                                                   double dbgain,
                                                                   double bw_norm,
                                                                   std::size_t wor_n,
                                                                   bool whole) {
    const double A = std::pow(10.0, dbgain / 20.0);
    const double Az = dbgain > 0.0 ? A : 1.0;
    const double Ap = dbgain > 0.0 ? 1.0 : A;
    const double q_value = 1.0 / (2.0 * std::sinh((std::log(2.0) / 2.0) * bw_norm));
    const std::vector<double> b{1.0, Az / q_value, 1.0};
    const std::vector<double> a{1.0, 1.0 / (Ap * q_value), 1.0};
    return zpk_freqz_cpp(ba_lp2lp_bilinear_cpp(b, a, wn_norm, false), wor_n, whole);
}

static std::vector<std::complex<double>> zpk_freqz_cpp(const DigitalZpk& filt,
                                                       std::size_t wor_n,
                                                       bool whole) {
    std::vector<std::complex<double>> out(wor_n);
    const double step = whole ? (TAU / static_cast<double>(wor_n))
                              : ((wor_n > 1U) ? (PI / static_cast<double>(wor_n - 1U)) : 0.0);
    for (std::size_t i = 0; i < wor_n; ++i) {
        const double w = step * static_cast<double>(i);
        const std::complex<double> z = std::exp(std::complex<double>(0.0, w));
        std::complex<double> num = filt.k;
        for (const auto& zero : filt.z) num *= (z - zero);
        std::complex<double> den{1.0, 0.0};
        for (const auto& pole : filt.p) den *= (z - pole);
        out[i] = num / den;
    }
    return out;
}

static DigitalZpk butter_digital_lowpass_zpk_cpp(int order, double cutoff_hz, double fs) {
    return bilinear_zpk_cpp(lp2lp_zpk_cpp(buttap_zpk_cpp(order), 2.0 * fs * std::tan(PI * cutoff_hz / fs)), fs);
}

static DigitalZpk butter_digital_highpass_zpk_cpp(int order, double cutoff_hz, double fs) {
    return bilinear_zpk_cpp(lp2hp_zpk_cpp(buttap_zpk_cpp(order), 2.0 * fs * std::tan(PI * cutoff_hz / fs)), fs);
}

static DigitalZpk butter_digital_bandpass_zpk_cpp(int order, double low_hz, double high_hz, double fs) {
    const double warped_low = 2.0 * fs * std::tan(PI * low_hz / fs);
    const double warped_high = 2.0 * fs * std::tan(PI * high_hz / fs);
    const double bw = warped_high - warped_low;
    const double wo = std::sqrt(warped_low * warped_high);
    return bilinear_zpk_cpp(lp2bp_zpk_cpp(buttap_zpk_cpp(order), wo, bw), fs);
}

static std::vector<std::complex<double>> poly_from_roots_cpp(
    const std::vector<std::complex<double>>& roots)
{
    std::vector<std::complex<double>> poly{std::complex<double>(1.0, 0.0)};
    for (const auto& r : roots) {
        std::vector<std::complex<double>> next(poly.size() + 1U, std::complex<double>(0.0, 0.0));
        for (std::size_t i = 0; i < poly.size(); ++i) {
            next[i] += poly[i];
            next[i + 1U] -= poly[i] * r;
        }
        poly.swap(next);
    }
    return poly;
}

static void zpk_to_ba_cpp(const DigitalZpk& zpk,
                          std::vector<double>& b,
                          std::vector<double>& a)
{
    auto b_poly = poly_from_roots_cpp(zpk.z);
    auto a_poly = poly_from_roots_cpp(zpk.p);
    for (auto& coeff : b_poly) coeff *= zpk.k;

    b.resize(b_poly.size());
    a.resize(a_poly.size());
    for (std::size_t i = 0; i < b_poly.size(); ++i) b[i] = b_poly[i].real();
    for (std::size_t i = 0; i < a_poly.size(); ++i) a[i] = a_poly[i].real();

    if (!a.empty() && a[0] != 1.0) {
        const double norm = a[0];
        for (double& v : b) v /= norm;
        for (double& v : a) v /= norm;
    }
}

static std::vector<double> solve_linear_system_cpp(std::vector<double> mat,
                                                   std::vector<double> rhs,
                                                   int n)
{
    for (int col = 0; col < n; ++col) {
        int pivot = col;
        double best = std::fabs(mat[(size_t)col * n + col]);
        for (int row = col + 1; row < n; ++row) {
            double cand = std::fabs(mat[(size_t)row * n + col]);
            if (cand > best) {
                best = cand;
                pivot = row;
            }
        }
        if (pivot != col) {
            for (int k = col; k < n; ++k) {
                std::swap(mat[(size_t)col * n + k], mat[(size_t)pivot * n + k]);
            }
            std::swap(rhs[col], rhs[pivot]);
        }
        double diag = mat[(size_t)col * n + col];
        if (std::fabs(diag) < 1e-18) continue;
        for (int k = col; k < n; ++k) mat[(size_t)col * n + k] /= diag;
        rhs[col] /= diag;
        for (int row = 0; row < n; ++row) {
            if (row == col) continue;
            double factor = mat[(size_t)row * n + col];
            if (factor == 0.0) continue;
            for (int k = col; k < n; ++k) {
                mat[(size_t)row * n + k] -= factor * mat[(size_t)col * n + k];
            }
            rhs[row] -= factor * rhs[col];
        }
    }
    return rhs;
}

static std::vector<double> lfilter_zi_cpp(const std::vector<double>& b_in,
                                          const std::vector<double>& a_in)
{
    const int n = static_cast<int>(std::max(b_in.size(), a_in.size()));
    const int m = n - 1;
    if (m <= 0) return {};

    std::vector<double> b(n, 0.0), a(n, 0.0);
    std::copy(b_in.begin(), b_in.end(), b.begin());
    std::copy(a_in.begin(), a_in.end(), a.begin());

    std::vector<double> mat((size_t)m * m, 0.0);
    std::vector<double> rhs(m, 0.0);
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < m; ++col) {
            double A = 0.0;
            if (col == 0) A = -a[row + 1];
            else if (row == col - 1) A = 1.0;
            mat[(size_t)row * m + col] = ((row == col) ? 1.0 : 0.0) - A;
        }
        rhs[row] = b[row + 1] - a[row + 1] * b[0];
    }
    return solve_linear_system_cpp(std::move(mat), std::move(rhs), m);
}

static void lfilter_df2t_cpp(const std::vector<double>& b_in,
                             const std::vector<double>& a_in,
                             const std::vector<double>& x,
                             std::vector<double>& y,
                             std::vector<double> z)
{
    const int n = static_cast<int>(std::max(b_in.size(), a_in.size()));
    const int m = n - 1;
    std::vector<double> b(n, 0.0), a(n, 0.0);
    std::copy(b_in.begin(), b_in.end(), b.begin());
    std::copy(a_in.begin(), a_in.end(), a.begin());
    if (static_cast<int>(z.size()) < m) z.resize(m, 0.0);
    y.resize(x.size());

    for (std::size_t i = 0; i < x.size(); ++i) {
        const double xi = x[i];
        double yi = b[0] * xi;
        if (m > 0) yi += z[0];
        for (int j = 0; j < m - 1; ++j) {
            z[j] = b[j + 1] * xi + z[j + 1] - a[j + 1] * yi;
        }
        if (m > 0) z[m - 1] = b[m] * xi - a[m] * yi;
        y[i] = yi;
    }
}

static std::vector<double> odd_extend_cpp(const double* x, int n, int edge)
{
    std::vector<double> out((size_t)n + 2U * (size_t)edge);
    for (int i = 0; i < edge; ++i) {
        out[(size_t)i] = 2.0 * x[0] - x[edge - i];
    }
    std::memcpy(out.data() + edge, x, (size_t)n * sizeof(double));
    for (int i = 0; i < edge; ++i) {
        out[(size_t)edge + (size_t)n + (size_t)i] = 2.0 * x[n - 1] - x[n - 2 - i];
    }
    return out;
}

static void filtfilt_ba_cpp(const std::vector<double>& b,
                            const std::vector<double>& a,
                            const double* x,
                            int n,
                            std::vector<double>& y)
{
    const int edge = 3 * static_cast<int>(std::max(a.size(), b.size()) - 1);
    auto ext = odd_extend_cpp(x, n, edge);
    auto zi = lfilter_zi_cpp(b, a);
    std::vector<double> zi_scaled = zi;
    for (double& v : zi_scaled) v *= ext[0];
    lfilter_df2t_cpp(b, a, ext, y, zi_scaled);
    std::reverse(y.begin(), y.end());
    zi_scaled = zi;
    for (double& v : zi_scaled) v *= y[0];
    std::vector<double> y2;
    lfilter_df2t_cpp(b, a, y, y2, zi_scaled);
    std::reverse(y2.begin(), y2.end());
    y.assign(y2.begin() + edge, y2.begin() + edge + n);
}

// ============================================================
// CUDA Kernels
// ============================================================

__global__ void k_copy_and_pad(
    const double* lead_in, int lead_count, const double* src, const double* tail_in, int tail_count,
    double* dst, int samples_per_field, int fft_size, 
    int total_dst_samples, int total_src_samples, int overlap)       
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_dst_samples) return;
    
    int field = idx / fft_size;
    int pos   = idx % fft_size;

    // OVERLAP-SAVE: Shift read window back into the past
    long long src_idx = (long long)field * samples_per_field + pos - overlap;

    if (src_idx < 0) {
        const int lead_idx = lead_count + static_cast<int>(src_idx);
        dst[idx] = (lead_in && lead_idx >= 0) ? lead_in[lead_idx] : src[0];
    } else if (src_idx < total_src_samples) {
        dst[idx] = src[src_idx];
    } else {
        const int tail_idx = static_cast<int>(src_idx - total_src_samples);
        dst[idx] = (tail_in && tail_idx < tail_count) ? tail_in[tail_idx] : src[total_src_samples - 1];
    }
}

__global__ void k_trim(
    const double* src, double* dst, int samples_per_field, int fft_size, 
    int total_dst_samples, int overlap)       
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_dst_samples) return;
    int field = idx / samples_per_field;
    int pos   = idx % samples_per_field;
    
    // OVERLAP-SAVE: Shift write window forward to discard the trash zone
    int src_idx = pos + overlap;
    if (src_idx >= fft_size) src_idx = fft_size - 1; 
    
    dst[idx] = src[(size_t)field * fft_size + src_idx];
}

__global__ void k_trim_complex(
    const cufftDoubleComplex* src, cufftDoubleComplex* dst, int samples_per_field, int fft_size,
    int total_dst_samples, int overlap)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_dst_samples) return;
    int field = idx / samples_per_field;
    int pos = idx % samples_per_field;
    int src_idx = pos + overlap;
    if (src_idx >= fft_size) src_idx = fft_size - 1;
    dst[idx] = src[static_cast<size_t>(field) * fft_size + src_idx];
}

__global__ void k_roll_fields_left(const double* src, double* dst, int spf, int total_fields, int shift) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = total_fields * spf;
    if (idx >= total) return;
    int field = idx / spf;
    int pos = idx % spf;
    int src_pos = pos + shift;
    if (src_pos >= spf) src_pos -= spf;
    dst[idx] = src[field * spf + src_pos];
}

__global__ void k_diff_analytic_inplace(cufftDoubleComplex* analytic, int fft_size, int total_fields) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = total_fields * fft_size;
    if (idx >= total) return;
    int pos = idx % fft_size;
    cufftDoubleComplex out{};
    if (pos != 0) {
        cufftDoubleComplex cur = analytic[idx];
        cufftDoubleComplex prev = analytic[idx - 1];
        out.x = cur.x - prev.x;
        out.y = cur.y - prev.y;
    }
    analytic[idx] = out;
}

__global__ void k_replace_spikes(double* demod, const double* demod_diffed, int spf, int total_fields,
                                 double max_value, int replace_start, int replace_end) {
    int field = blockIdx.x * blockDim.x + threadIdx.x;
    if (field >= total_fields) return;
    double* d = demod + static_cast<size_t>(field) * spf;
    const double* dd = demod_diffed + static_cast<size_t>(field) * spf;
    for (int i = 0; i < spf; ++i) {
        if (d[i] <= max_value) continue;
        int start = i - replace_start;
        if (start < 0) start = 0;
        int end = i + replace_end;
        if (end > spf - 1) end = spf - 1;
        double max_a = d[start];
        double max_b = dd[start];
        for (int j = start + 1; j < end; ++j) {
            if (d[j] > max_a) max_a = d[j];
            if (dd[j] > max_b) max_b = dd[j];
        }
        if (max_b < max_a) {
            for (int j = start; j < end; ++j) d[j] = dd[j];
        }
    }
}

__global__ void k_apply_rf_filter(
    cufftDoubleComplex* fft_data, const double* rf_filter, int freq_bins, int total_bins)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_bins) return;
    int bin = idx % freq_bins;
    double rf = rf_filter[bin];
    fft_data[idx].x *= rf;
    fft_data[idx].y *= rf;
}

__global__ void k_expand_half_to_full(
    const cufftDoubleComplex* half, cufftDoubleComplex* full, int freq_bins, int fft_size, int num_fields)
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

__global__ void k_compute_angles(
    const cufftDoubleComplex* analytic, double* angles, int total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    angles[idx] = atan2(analytic[idx].y, analytic[idx].x);
}

__global__ void k_compute_envelope(
    const cufftDoubleComplex* __restrict__ analytic, double* __restrict__ envelope, 
    int spf, int fft_size, int total_fields, int overlap)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = total_fields * spf;
    if (idx >= total) return;
    int field = idx / spf;
    int sample = idx % spf;
    
    // OVERLAP-SAVE: Realign envelope read
    int src = field * fft_size + sample + overlap; 
    envelope[idx] = fabs(analytic[src].x) / (static_cast<double>(fft_size) * 256.0);
}

__global__ void k_compute_envelope_block(
    const double* __restrict__ filtered, double* __restrict__ envelope,
    int spf, int total_fields)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = total_fields * spf;
    if (idx >= total) return;
    envelope[idx] = fabs(filtered[idx]) / (static_cast<double>(spf) * 256.0);
}

__global__ void k_roll_and_remove_dc(
    const double* __restrict__ src,
    double* __restrict__ dst,
    int spf,
    int total_fields,
    int roll_shift)
{
    const int field = blockIdx.x;
    if (field >= total_fields) return;

    __shared__ double mean_value;
    if (threadIdx.x == 0) {
        double sum = 0.0;
        const double* field_src = src + static_cast<size_t>(field) * spf;
        for (int i = 0; i < spf; ++i) sum += field_src[i];
        mean_value = sum / static_cast<double>(spf);
    }
    __syncthreads();

    const double* field_src = src + static_cast<size_t>(field) * spf;
    double* field_dst = dst + static_cast<size_t>(field) * spf;
    for (int i = threadIdx.x; i < spf; i += blockDim.x) {
        int src_idx = i - roll_shift;
        while (src_idx < 0) src_idx += spf;
        while (src_idx >= spf) src_idx -= spf;
        field_dst[i] = (field_src[src_idx] - mean_value) / 256.0;
    }
}

__device__ inline double odd_ext_value(const double* x, int n, int edge, int idx) {
    if (idx < edge) {
        return (2.0 * x[0]) - x[edge - idx];
    }
    if (idx < edge + n) {
        return x[idx - edge];
    }
    const int j = idx - (edge + n);
    return (2.0 * x[n - 1]) - x[n - 2 - j];
}

__device__ inline double rolled_sample(const double* x, int n, int roll_shift, int pos) {
    int src = pos - roll_shift;
    while (src < 0) src += n;
    while (src >= n) src -= n;
    return x[src];
}

__device__ inline double odd_ext_value_rolled(const double* x, int n, int edge, int idx, int roll_shift) {
    if (idx < edge) {
        return (2.0 * rolled_sample(x, n, roll_shift, 0)) -
               rolled_sample(x, n, roll_shift, edge - idx);
    }
    if (idx < edge + n) {
        return rolled_sample(x, n, roll_shift, idx - edge);
    }
    const int j = idx - (edge + n);
    return (2.0 * rolled_sample(x, n, roll_shift, n - 1)) -
           rolled_sample(x, n, roll_shift, n - 2 - j);
}

__global__ void k_roll_and_filtfilt_env_first_order(
    const double* __restrict__ raw_env, double* __restrict__ envelope,
    double* __restrict__ work, int spf, int total_fields,
    double b0, double b1, double a1, double zi_base, int roll_shift)
{
    const int field = blockIdx.x * blockDim.x + threadIdx.x;
    if (field >= total_fields) return;

    constexpr int edge = 6;
    const int ext_len = spf + 2 * edge;
    const double* x = raw_env + static_cast<size_t>(field) * spf;
    double* y_ext = work + static_cast<size_t>(field) * ext_len;

    for (int i = 0; i < ext_len; ++i) {
        y_ext[i] = odd_ext_value_rolled(x, spf, edge, i, roll_shift);
    }

    double z0 = zi_base * y_ext[0];
    for (int i = 0; i < ext_len; ++i) {
        const double in = y_ext[i];
        const double out = b0 * in + z0;
        z0 = b1 * in - a1 * out;
        y_ext[i] = out;
    }

    for (int i = 0, j = ext_len - 1; i < j; ++i, --j) {
        const double tmp = y_ext[i];
        y_ext[i] = y_ext[j];
        y_ext[j] = tmp;
    }

    z0 = zi_base * y_ext[0];
    for (int i = 0; i < ext_len; ++i) {
        const double in = y_ext[i];
        const double out = b0 * in + z0;
        z0 = b1 * in - a1 * out;
        y_ext[i] = out;
    }

    for (int i = 0, j = ext_len - 1; i < j; ++i, --j) {
        const double tmp = y_ext[i];
        y_ext[i] = y_ext[j];
        y_ext[j] = tmp;
    }

    for (int i = 0; i < spf; ++i) {
        envelope[static_cast<size_t>(field) * spf + i] = y_ext[edge + i];
    }
}

__global__ void k_unwrap_to_hz(
    const double* angles, double* demod, int fft_size, int samples_per_field, 
    double freq_hz, int num_fields, int overlap)
{
    int field = blockIdx.x * blockDim.x + threadIdx.x;
    if (field >= num_fields) return;

    const double* a = angles + (size_t)field * fft_size;
    double* out = demod + (size_t)field * samples_per_field;

    double scale = freq_hz / TAU;

    for (int i = 0; i < samples_per_field; i++) {
        int a_idx = overlap + i;
        if (a_idx <= 0) {
            out[i] = 0.0;
            continue;
        }
        double delta = a[a_idx] - a[a_idx - 1];

        while (delta < 0.0) delta += TAU;
        while (delta > TAU) delta -= TAU;
        
        out[i] = delta * scale;
    }
}

__global__ void k_apply_complex_filter(
    const cufftDoubleComplex* in, cufftDoubleComplex* out, const cufftDoubleComplex* filter, 
    int freq_bins, int total_bins)
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
// Scratch memory budget
// ============================================================
size_t FMDemodState::scratch_bytes_per_field(int samples_per_field) {
    int fs = next_fft_size(samples_per_field + OVERLAP_SAMPLES);
    int fb = fs / 2 + 1;
    return (size_t)fb * sizeof(cufftDoubleComplex)     
         + (size_t)fs * sizeof(cufftDoubleComplex)     
         + (size_t)fb * sizeof(cufftDoubleComplex);    
}

// ============================================================
// Init
// ============================================================
bool FMDemodState::init(const VideoFormat& fmt, int max_batch_size, int samples_per_field_override) {
    max_batch = max_batch_size;
    int spf = (samples_per_field_override > 0) ? samples_per_field_override : fmt.samples_per_field;
    
    // ALLOCATE SPACE FOR THE TRASH ZONE
    fft_size  = next_fft_size(spf + OVERLAP_SAMPLES);
    freq_bins = fft_size / 2 + 1;
    env_fft_size = spf;
    env_freq_bins = (env_fft_size / 2) + 1;

    fprintf(stderr, "  [FM demod] FFT size: %d (field: %d, pad: +%d)  freq bins: %d  batch: %d\n",
            fft_size, spf, fft_size - spf, freq_bins, max_batch);

    double fs      = fmt.sample_rate;
    double nyquist = fs / 2.0;

    double bpf_low          = 1300000.0;
    double bpf_high         = 7500000.0;
    int    bpf_order        = 8;
    double lpf_extra_freq   = 8000000.0;
    int    lpf_extra_ord    = 8;
    double hpf_extra_freq   = 1000000.0;
    int    hpf_extra_ord    = 8;
    double video_lpf_freq   = 3000000.0;
    int    video_lpf_order  = 9;
    bool   use_rf_ramp      = false;
    double ramp_start_freq  = 0.0;
    double ramp_boost_0     = 0.0;
    double ramp_boost_20    = 1.0;
    bool   ramp_double      = false;
    bool   use_rf_peak      = false;
    double rf_peak_freq     = 0.0;
    double rf_peak_gain     = 0.0;
    double rf_peak_bw       = 0.0;

    if (fmt.profile == VideoProfile::NTSC_525_60_VHS || fmt.profile == VideoProfile::MPAL_525_60_VHS) {
        bpf_low = 500000.0;
        bpf_high = (fmt.tape_speed == TapeSpeed::EP) ? 6400000.0 : 6500000.0;
        bpf_order = 8;
        lpf_extra_freq = (fmt.tape_speed == TapeSpeed::SP) ? 6000000.0 :
                         (fmt.tape_speed == TapeSpeed::LP) ? 5900000.0 : 5800000.0;
        lpf_extra_ord = 25;
        hpf_extra_freq = 1200000.0;
        hpf_extra_ord = 20;
        video_lpf_freq = 6600000.0;
        video_lpf_order = 9;
        use_rf_ramp = true;
        ramp_start_freq = fmt.chroma_under;
        ramp_boost_0 = 0.0;
        ramp_boost_20 = 1.0;
        ramp_double = false;
        use_rf_peak = true;
        rf_peak_freq = 3900000.0;
        rf_peak_gain = 4.0;
        rf_peak_bw = 10000000.0;
    }

    const int reference_spf = 32768;
    const int reference_fft_size = next_fft_size(reference_spf + OVERLAP_SAMPLES);
    const int reference_freq_bins = reference_fft_size / 2 + 1;
    const int reference_env_fft_size = reference_spf;
    const int reference_env_freq_bins = reference_env_fft_size / 2 + 1;

    std::vector<double> h_rf(freq_bins);
    std::vector<double> env_h_rf(env_freq_bins);
    std::vector<double> h_burst(env_freq_bins);

    std::vector<double> ref_rf(reference_freq_bins);
    std::vector<double> ref_env_rf(reference_env_freq_bins);
    std::vector<double> ref_burst(reference_env_freq_bins);

    std::vector<double> rf_bpf_mag(reference_freq_bins, 1.0);
    std::vector<double> env_rf_bpf_mag(reference_env_freq_bins, 1.0);
    if (bpf_order > 0) {
        auto rf_bpf = zpk_freqz_cpp(
            butter_digital_bandpass_zpk_cpp(bpf_order, bpf_low, bpf_high, fs),
            static_cast<std::size_t>(reference_fft_size),
            true);
        for (int k = 0; k < reference_freq_bins; ++k) {
            rf_bpf_mag[static_cast<std::size_t>(k)] = std::abs(rf_bpf[static_cast<std::size_t>(k)]);
        }
        auto env_rf_bpf = zpk_freqz_cpp(
            butter_digital_bandpass_zpk_cpp(bpf_order, bpf_low, bpf_high, fs),
            static_cast<std::size_t>(reference_env_fft_size),
            true);
        for (int k = 0; k < reference_env_freq_bins; ++k) {
            env_rf_bpf_mag[static_cast<std::size_t>(k)] = std::abs(env_rf_bpf[static_cast<std::size_t>(k)]);
        }
    }
    auto rf_lpf = zpk_freqz_cpp(
        butter_digital_lowpass_zpk_cpp(lpf_extra_ord, lpf_extra_freq, fs),
        static_cast<std::size_t>(reference_fft_size),
        true);
    auto rf_hpf = zpk_freqz_cpp(
        butter_digital_highpass_zpk_cpp(hpf_extra_ord, hpf_extra_freq, fs),
        static_cast<std::size_t>(reference_fft_size),
        true);
    auto env_rf_lpf = zpk_freqz_cpp(
        butter_digital_lowpass_zpk_cpp(lpf_extra_ord, lpf_extra_freq, fs),
        static_cast<std::size_t>(reference_env_fft_size),
        true);
    auto env_rf_hpf = zpk_freqz_cpp(
        butter_digital_highpass_zpk_cpp(hpf_extra_ord, hpf_extra_freq, fs),
        static_cast<std::size_t>(reference_env_fft_size),
        true);
    std::vector<std::complex<double>> rf_peak_resp;
    std::vector<std::complex<double>> env_rf_peak_resp;
    if (use_rf_peak) {
        rf_peak_resp = peaking_freq_response_cpp(
            rf_peak_freq / nyquist, rf_peak_gain, rf_peak_bw / nyquist,
            static_cast<std::size_t>(reference_fft_size), true);
        env_rf_peak_resp = peaking_freq_response_cpp(
            rf_peak_freq / nyquist, rf_peak_gain, rf_peak_bw / nyquist,
            static_cast<std::size_t>(reference_env_fft_size), true);
    }
    for (int k = 0; k < reference_freq_bins; ++k) {
        double f_hz = (double)k * nyquist / (double)(reference_freq_bins - 1);
        double bpf = rf_bpf_mag[static_cast<std::size_t>(k)];
        double lpf = std::abs(rf_lpf[static_cast<std::size_t>(k)]);
        double hpf = std::abs(rf_hpf[static_cast<std::size_t>(k)]);
        double peak = use_rf_peak ? std::abs(rf_peak_resp[static_cast<std::size_t>(k)]) : 1.0;
        double ramp = 1.0;
        if (use_rf_ramp) {
            ramp = ramp_filter_mag(f_hz, ramp_start_freq, ramp_boost_0, ramp_boost_20, nyquist);
            if (ramp_double) ramp *= ramp;
        }
        double hil = (k == 0 || k == reference_freq_bins - 1) ? 1.0 : 2.0;
        ref_rf[static_cast<std::size_t>(k)] = bpf * lpf * hpf * peak * ramp * hil;
    }
    for (int k = 0; k < reference_env_freq_bins; ++k) {
        double f_hz = (double)k * nyquist / (double)(reference_env_freq_bins - 1);
        double bpf = env_rf_bpf_mag[static_cast<std::size_t>(k)];
        double lpf = std::abs(env_rf_lpf[static_cast<std::size_t>(k)]);
        double hpf = std::abs(env_rf_hpf[static_cast<std::size_t>(k)]);
        double peak = use_rf_peak ? std::abs(env_rf_peak_resp[static_cast<std::size_t>(k)]) : 1.0;
        double ramp = 1.0;
        if (use_rf_ramp) {
            ramp = ramp_filter_mag(f_hz, ramp_start_freq, ramp_boost_0, ramp_boost_20, nyquist);
            if (ramp_double) ramp *= ramp;
        }
        ref_env_rf[static_cast<std::size_t>(k)] = bpf * lpf * hpf * peak * ramp;
    }

    const double burst_low_hz = 60000.0;
    const double burst_high_hz = (fmt.profile == VideoProfile::PAL_625_50_VHS) ? 1500000.0 : 1200000.0;
    auto burst_bpf = zpk_freqz_cpp(
        butter_digital_bandpass_zpk_cpp(4, burst_low_hz, burst_high_hz, fs),
        static_cast<std::size_t>(reference_env_fft_size),
        true);
    for (int k = 0; k < reference_env_freq_bins; ++k) {
        ref_burst[static_cast<std::size_t>(k)] = std::norm(burst_bpf[static_cast<std::size_t>(k)]);
    }

    h_rf = interp_real_cpp(ref_rf, static_cast<std::size_t>(freq_bins));
    env_h_rf = interp_real_cpp(ref_env_rf, static_cast<std::size_t>(env_freq_bins));
    h_burst = interp_real_cpp(ref_burst, static_cast<std::size_t>(env_freq_bins));
    for (double& v : h_burst) {
        v /= static_cast<double>(env_fft_size);
    }

    double deemph_mid  = 273755.82;
    double deemph_gain = 13.9794;
    double deemph_q    = 0.462088186;

    double sync_cutoff_norm = 500000.0 / nyquist;
    auto   sync_fir = firwin_lpf(65, sync_cutoff_norm);
    f05_offset = 32;
    first_order_lowpass_coeffs(fs, 700000.0, env_b0, env_b1, env_a1, env_zi);

    std::vector<cufftDoubleComplex> h_fv(freq_bins), h_fv05(freq_bins);
    const auto filter_deemp_ref = freqz_ba_cpp(
        make_fm_deemphasis_b_cpp(fs, deemph_gain, deemph_mid, deemph_q),
        static_cast<std::size_t>(reference_freq_bins),
        false);
    std::vector<std::complex<double>> ref_fv(reference_freq_bins);
    std::vector<std::complex<double>> ref_fv05(reference_freq_bins);
    for (int k = 0; k < reference_freq_bins; ++k) {
        double omega = PI * (double)k / (double)(reference_freq_bins - 1);
        double f_hz = (double)k * nyquist / (double)(reference_freq_bins - 1);
        std::complex<double> deemp = filter_deemp_ref[static_cast<std::size_t>(k)];
        double vlpf = supergauss_mag(f_hz, video_lpf_freq, video_lpf_order);
        std::complex<double> sync = fir_freqz(sync_fir, omega, 0);
        ref_fv[static_cast<std::size_t>(k)] = deemp * vlpf;
        ref_fv05[static_cast<std::size_t>(k)] = deemp * vlpf * sync;
    }
    const double inv_n = 1.0 / (double)fft_size;
    auto fv_interp = interp_complex_cpp(ref_fv, static_cast<std::size_t>(freq_bins));
    auto fv05_interp = interp_complex_cpp(ref_fv05, static_cast<std::size_t>(freq_bins));
    for (int k = 0; k < freq_bins; ++k) {
        const std::complex<double> fv = fv_interp[static_cast<std::size_t>(k)] * inv_n;
        const std::complex<double> fv05 = fv05_interp[static_cast<std::size_t>(k)] * inv_n;
        h_fv[k].x = fv.real();
        h_fv[k].y = fv.imag();
        h_fv05[k].x = fv05.real();
        h_fv05[k].y = fv05.imag();
    }

    cudaMalloc(&d_rf_filter, freq_bins * sizeof(double));
    cudaMemcpy(d_rf_filter, h_rf.data(), freq_bins * sizeof(double), cudaMemcpyHostToDevice);
    cudaMalloc(&d_env_rf_filter, env_freq_bins * sizeof(double));
    cudaMemcpy(d_env_rf_filter, env_h_rf.data(), env_freq_bins * sizeof(double), cudaMemcpyHostToDevice);
    cudaMalloc(&d_burst_filter, env_freq_bins * sizeof(double));
    cudaMemcpy(d_burst_filter, h_burst.data(), env_freq_bins * sizeof(double), cudaMemcpyHostToDevice);

    cudaMalloc(&d_fvideo, freq_bins * sizeof(cufftDoubleComplex));
    cudaMemcpy(d_fvideo, h_fv.data(), freq_bins * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice);

    cudaMalloc(&d_fvideo05, freq_bins * sizeof(cufftDoubleComplex));
    cudaMemcpy(d_fvideo05, h_fv05.data(), freq_bins * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice);

    cudaMalloc(&d_fft_half, (size_t)max_batch * freq_bins * sizeof(cufftDoubleComplex));
    cudaMalloc(&d_analytic, (size_t)max_batch * fft_size  * sizeof(cufftDoubleComplex));
    cudaMalloc(&d_post_fft, (size_t)max_batch * freq_bins * sizeof(cufftDoubleComplex));
    cudaMalloc(&d_env_work, (size_t)max_batch * (spf + 12) * sizeof(double));
    cudaMalloc(&d_angle_work, (size_t)max_batch * fs * sizeof(double));
    cudaMalloc(&d_kept_analytic, (size_t)max_batch * spf * sizeof(cufftDoubleComplex));
    cudaMalloc(&d_env_fft_half, (size_t)max_batch * env_freq_bins * sizeof(cufftDoubleComplex));
    cudaMalloc(&d_env_filtered, (size_t)max_batch * env_fft_size * sizeof(double));

    {
        int n[] = { fft_size };
        cufftResult r = cufftPlanMany(&plan_r2c, 1, n, NULL, 1, fft_size, NULL, 1, freq_bins, CUFFT_D2Z, max_batch);
        if (r != CUFFT_SUCCESS) return false;
    }
    {
        int n[] = { fft_size };
        cufftResult r = cufftPlanMany(&plan_c2c_inv, 1, n, NULL, 1, fft_size, NULL, 1, fft_size, CUFFT_Z2Z, max_batch);
        if (r != CUFFT_SUCCESS) return false;
    }
    {
        int n[] = { fft_size };
        cufftResult r = cufftPlanMany(&plan_c2r, 1, n, NULL, 1, freq_bins, NULL, 1, fft_size, CUFFT_Z2D, max_batch);
        if (r != CUFFT_SUCCESS) return false;
    }
    {
        int n[] = { env_fft_size };
        cufftResult r = cufftPlanMany(&plan_env_r2c, 1, n, NULL, 1, env_fft_size, NULL, 1, env_freq_bins, CUFFT_D2Z, max_batch);
        if (r != CUFFT_SUCCESS) return false;
    }
    {
        int n[] = { env_fft_size };
        cufftResult r = cufftPlanMany(&plan_env_c2r, 1, n, NULL, 1, env_freq_bins, NULL, 1, env_fft_size, CUFFT_Z2D, max_batch);
        if (r != CUFFT_SUCCESS) return false;
    }

    return true;
}

void FMDemodState::destroy() {
    if (plan_r2c)     { cufftDestroy(plan_r2c);     plan_r2c = 0; }
    if (plan_c2c_inv) { cufftDestroy(plan_c2c_inv); plan_c2c_inv = 0; }
    if (plan_c2r)     { cufftDestroy(plan_c2r);     plan_c2r = 0; }
    if (plan_env_r2c) { cufftDestroy(plan_env_r2c); plan_env_r2c = 0; }
    if (plan_env_c2r) { cufftDestroy(plan_env_c2r); plan_env_c2r = 0; }

    if (d_rf_filter) { cudaFree(d_rf_filter); d_rf_filter = nullptr; }
    if (d_env_rf_filter) { cudaFree(d_env_rf_filter); d_env_rf_filter = nullptr; }
    if (d_burst_filter) { cudaFree(d_burst_filter); d_burst_filter = nullptr; }
    if (d_fvideo)    { cudaFree(d_fvideo);    d_fvideo    = nullptr; }
    if (d_fvideo05)  { cudaFree(d_fvideo05);  d_fvideo05  = nullptr; }
    if (d_fft_half)  { cudaFree(d_fft_half);  d_fft_half  = nullptr; }
    if (d_analytic)  { cudaFree(d_analytic);  d_analytic  = nullptr; }
    if (d_post_fft)  { cudaFree(d_post_fft);  d_post_fft  = nullptr; }
    if (d_env_work)  { cudaFree(d_env_work);  d_env_work  = nullptr; }
    if (d_angle_work){ cudaFree(d_angle_work); d_angle_work = nullptr; }
    if (d_kept_analytic) { cudaFree(d_kept_analytic); d_kept_analytic = nullptr; }
    if (d_env_fft_half) { cudaFree(d_env_fft_half); d_env_fft_half = nullptr; }
    if (d_env_filtered) { cudaFree(d_env_filtered); d_env_filtered = nullptr; }
}

bool fm_demod_init(FMDemodState& state, const VideoFormat& fmt, int max_batch) {
    return state.init(fmt, max_batch);
}

// ============================================================
// FM Demodulation Main Loop
// ============================================================
void fm_demod(FMDemodState& state,
              const double* d_lead_in, const double* d_raw, const double* d_tail_in,
              double* d_demod, double* d_demod_05, double* d_demod_burst, double* d_envelope,
              double* d_debug_raw_env, double* d_debug_demod_raw, double* d_debug_demod_diff,
              double* d_debug_demod_spikefixed, cufftDoubleComplex* d_debug_hilbert,
              int num_fields, size_t samples_per_field, const VideoFormat& fmt)
{
    int fb  = state.freq_bins;
    int fs  = state.fft_size;
    int spf = (int)samples_per_field;
    int T   = 256;   

    double* d_scratch = reinterpret_cast<double*>(state.d_analytic);

    // ========== PART A: RF filtering + analytic signal ==========
    {
        int total_dst = num_fields * fs;
        int total_src = num_fields * spf; 
        int blocks = (total_dst + T - 1) / T;
        k_copy_and_pad<<<blocks, T>>>(d_lead_in, OVERLAP_SAMPLES, d_raw, d_tail_in, OVERLAP_SAMPLES,
                                      d_scratch, spf, fs, total_dst, total_src, OVERLAP_SAMPLES);
    }

    cufftExecD2Z(state.plan_r2c, reinterpret_cast<cufftDoubleReal*>(d_scratch), state.d_fft_half);

    {
        int total = num_fields * fb;
        int blocks = (total + T - 1) / T;
        k_apply_rf_filter<<<blocks, T>>>(state.d_fft_half, state.d_rf_filter, fb, total);
    }

    {
        int total = num_fields * fs;
        int blocks = (total + T - 1) / T;
        k_expand_half_to_full<<<blocks, T>>>(state.d_fft_half, state.d_analytic, fb, fs, num_fields);
    }

    cufftExecZ2Z(state.plan_c2c_inv, state.d_analytic, state.d_analytic, CUFFT_INVERSE);

    if (d_envelope) {
        cufftExecD2Z(state.plan_env_r2c, const_cast<cufftDoubleReal*>(reinterpret_cast<const cufftDoubleReal*>(d_raw)), state.d_env_fft_half);
        {
            int total = num_fields * state.env_freq_bins;
            int blocks = (total + T - 1) / T;
            k_apply_rf_filter<<<blocks, T>>>(state.d_env_fft_half, state.d_env_rf_filter, state.env_freq_bins, total);
        }
        {
            int total = num_fields * state.env_fft_size;
            int blocks = (total + T - 1) / T;
            cufftExecZ2D(state.plan_env_c2r, state.d_env_fft_half, reinterpret_cast<cufftDoubleReal*>(state.d_env_filtered));
            k_compute_envelope_block<<<blocks, T>>>(state.d_env_filtered, d_envelope, spf, num_fields);
        }
        int total = num_fields * spf;
        if (d_debug_raw_env) {
            int blocks = (total + T - 1) / T;
            k_roll_fields_left<<<blocks, T>>>(d_envelope, d_debug_raw_env, spf, num_fields, spf - 4);
        }
        int field_blocks = (num_fields + T - 1) / T;
        k_roll_and_filtfilt_env_first_order<<<field_blocks, T>>>(
            d_envelope, d_envelope, state.d_env_work, spf, num_fields,
            state.env_b0, state.env_b1, state.env_a1, state.env_zi, 4);
    }

    if (d_demod_burst) {
        const size_t total_samples = static_cast<size_t>(num_fields) * spf;
        cudaMemcpy(state.d_env_filtered, d_raw, total_samples * sizeof(double), cudaMemcpyDeviceToDevice);
        cufftExecD2Z(state.plan_env_r2c,
                     reinterpret_cast<cufftDoubleReal*>(state.d_env_filtered),
                     state.d_env_fft_half);
        {
            int total = num_fields * state.env_freq_bins;
            int blocks = (total + T - 1) / T;
            k_apply_rf_filter<<<blocks, T>>>(state.d_env_fft_half, state.d_burst_filter, state.env_freq_bins, total);
        }
        cufftExecZ2D(state.plan_env_c2r, state.d_env_fft_half,
                     reinterpret_cast<cufftDoubleReal*>(state.d_env_filtered));
        {
            int roll_shift = static_cast<int>(5.0 * (fmt.sample_rate / 40.0e6));
            k_roll_and_remove_dc<<<num_fields, 256>>>(
                state.d_env_filtered, d_demod_burst, spf, num_fields, roll_shift);
        }
    }

    {
        int total = num_fields * fs;
        int blocks = (total + T - 1) / T;
        k_compute_angles<<<blocks, T>>>(state.d_analytic, state.d_angle_work, total);
    }

    {
        int blocks = (num_fields + T - 1) / T;
        k_unwrap_to_hz<<<blocks, T>>>(state.d_angle_work, d_demod, fs, spf, fmt.sample_rate, num_fields, OVERLAP_SAMPLES);
    }

    {
        int total_trimmed = num_fields * spf;
        int blocks = (total_trimmed + T - 1) / T;
        k_trim_complex<<<blocks, T>>>(state.d_analytic, state.d_kept_analytic, spf, fs, total_trimmed, OVERLAP_SAMPLES);
        if (d_debug_hilbert) {
            cudaMemcpy(d_debug_hilbert, state.d_kept_analytic, static_cast<size_t>(total_trimmed) * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToDevice);
        }
        k_diff_analytic_inplace<<<blocks, T>>>(state.d_kept_analytic, spf, num_fields);
        k_compute_angles<<<blocks, T>>>(state.d_kept_analytic, state.d_angle_work, total_trimmed);
    }
    {
        int blocks = (num_fields + T - 1) / T;
        k_unwrap_to_hz<<<blocks, T>>>(state.d_angle_work, d_demod_05, spf, spf, fmt.sample_rate, num_fields, 0);
        if (d_debug_demod_raw) {
            cudaMemcpy(d_debug_demod_raw, d_demod, static_cast<size_t>(num_fields) * spf * sizeof(double), cudaMemcpyDeviceToDevice);
        }
        if (d_debug_demod_diff) {
            cudaMemcpy(d_debug_demod_diff, d_demod_05, static_cast<size_t>(num_fields) * spf * sizeof(double), cudaMemcpyDeviceToDevice);
        }
        k_replace_spikes<<<blocks, T>>>(d_demod, d_demod_05, spf, num_fields, 8800000.0, 8, 30);
        if (d_debug_demod_spikefixed) {
            cudaMemcpy(d_debug_demod_spikefixed, d_demod, static_cast<size_t>(num_fields) * spf * sizeof(double), cudaMemcpyDeviceToDevice);
        }
    }

    // ========== PART B: post-demod filtering ==========
    {
        int total_dst = num_fields * fs;
        int total_src = num_fields * spf; 
        int blocks = (total_dst + T - 1) / T;
        k_copy_and_pad<<<blocks, T>>>(nullptr, 0, d_demod, nullptr, 0, d_scratch, spf, fs, total_dst, total_src, OVERLAP_SAMPLES);
    }

    cufftExecD2Z(state.plan_r2c, reinterpret_cast<cufftDoubleReal*>(d_scratch), state.d_fft_half);

    {
        int total = num_fields * fb;
        int blocks = (total + T - 1) / T;
        k_apply_complex_filter<<<blocks, T>>>(state.d_fft_half, state.d_post_fft, state.d_fvideo, fb, total);
    }
    
    cufftExecZ2D(state.plan_c2r, state.d_post_fft, reinterpret_cast<cufftDoubleReal*>(d_scratch));
    
    {
        int total_dst = num_fields * spf;
        int blocks = (total_dst + T - 1) / T;
        k_trim<<<blocks, T>>>(d_scratch, d_demod, spf, fs, total_dst, OVERLAP_SAMPLES);
    }

    {
        int total = num_fields * fb;
        int blocks = (total + T - 1) / T;
        k_apply_complex_filter<<<blocks, T>>>(state.d_fft_half, state.d_post_fft, state.d_fvideo05, fb, total);
    }
    
    cufftExecZ2D(state.plan_c2r, state.d_post_fft, reinterpret_cast<cufftDoubleReal*>(d_scratch));
    
    {
        int total_dst = num_fields * spf;
        int blocks = (total_dst + T - 1) / T;
        k_trim<<<blocks, T>>>(d_scratch, d_demod_05, spf, fs, total_dst, OVERLAP_SAMPLES);
        k_roll_fields_left<<<blocks, T>>>(d_demod_05, d_scratch, spf, num_fields, state.f05_offset);
        cudaMemcpy(d_demod_05, d_scratch, static_cast<size_t>(total_dst) * sizeof(double), cudaMemcpyDeviceToDevice);
    }

}

bool demod_burst_cpu(const double* raw,
                     int num_fields,
                     size_t samples_per_field,
                     const VideoFormat& fmt,
                     double* out)
{
    if (!raw || !out || num_fields <= 0 || samples_per_field == 0) return false;

    const double burst_low_hz = 60000.0;
    const double burst_high_hz = (fmt.profile == VideoProfile::PAL_625_50_VHS) ? 1500000.0 : 1200000.0;
    std::vector<double> b, a;
    zpk_to_ba_cpp(butter_digital_bandpass_zpk_cpp(4, burst_low_hz, burst_high_hz, fmt.sample_rate), b, a);

    const int spf = static_cast<int>(samples_per_field);
    const int move = static_cast<int>(5.0 * (fmt.sample_rate / 40.0e6));
    std::vector<double> filtered;
    std::vector<double> rolled((size_t)spf);

    for (int field = 0; field < num_fields; ++field) {
        const double* src = raw + (size_t)field * samples_per_field;
        filtfilt_ba_cpp(b, a, src, spf, filtered);
        for (int i = 0; i < spf; ++i) {
            int src_idx = i - move;
            while (src_idx < 0) src_idx += spf;
            while (src_idx >= spf) src_idx -= spf;
            rolled[(size_t)i] = filtered[(size_t)src_idx] / 256.0;
        }
        double mean = 0.0;
        for (double v : rolled) mean += v;
        mean /= static_cast<double>(spf);
        double* dst = out + (size_t)field * samples_per_field;
        for (int i = 0; i < spf; ++i) dst[i] = rolled[(size_t)i] - mean;
    }
    return true;
}

bool demod_burst_cpu_windowed(const double* raw_with_context,
                              size_t total_samples_with_context,
                              size_t lead_samples,
                              int num_fields,
                              size_t samples_per_field,
                              const VideoFormat& fmt,
                              double* out)
{
    if (!raw_with_context || !out || num_fields <= 0 || samples_per_field == 0) return false;
    if (total_samples_with_context == 0 || lead_samples >= total_samples_with_context) return false;

    const double burst_low_hz = 60000.0;
    const double burst_high_hz = (fmt.profile == VideoProfile::PAL_625_50_VHS) ? 1500000.0 : 1200000.0;
    std::vector<double> b, a;
    zpk_to_ba_cpp(butter_digital_bandpass_zpk_cpp(4, burst_low_hz, burst_high_hz, fmt.sample_rate), b, a);

    const int move = static_cast<int>(5.0 * (fmt.sample_rate / 40.0e6));
    std::vector<double> filtered;
    filtfilt_ba_cpp(b, a, raw_with_context, (int)total_samples_with_context, filtered);

    const size_t total_payload = (size_t)num_fields * samples_per_field;
    if (lead_samples + total_payload > total_samples_with_context) return false;

    std::vector<double> rolled(samples_per_field);
    for (int field = 0; field < num_fields; ++field) {
        const size_t field_base = lead_samples + (size_t)field * samples_per_field;
        for (size_t i = 0; i < samples_per_field; ++i) {
            long long src_idx = (long long)field_base + (long long)i - (long long)move;
            if (src_idx < 0) src_idx = 0;
            if ((size_t)src_idx >= total_samples_with_context) src_idx = (long long)total_samples_with_context - 1;
            rolled[i] = filtered[(size_t)src_idx] / 256.0;
        }
        double mean = 0.0;
        for (double v : rolled) mean += v;
        mean /= (double)samples_per_field;
        double* dst = out + (size_t)field * samples_per_field;
        for (size_t i = 0; i < samples_per_field; ++i) dst[i] = rolled[i] - mean;
    }
    return true;
}
