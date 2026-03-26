#include "pipeline/pipeline.h"
#include "pipeline/fm_demod.h"
#include "pipeline/sync_pulses.h"
#include "pipeline/line_locs.h"
#include "pipeline/hsync_refine.h"
#include "pipeline/tbc_resample.h"
#include "pipeline/chroma_decode.h"
#include "pipeline/dropout_detect.h"

#include <cuda_runtime.h>
#include <cufft.h>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <cmath>
#include <cstdlib>

// Debug: dump first N samples of a GPU buffer to /tmp for comparison with Python
static void dump_gpu_doubles(const char* path, const double* d_ptr, size_t count) {
    auto* h = new double[count];
    cudaMemcpy(h, d_ptr, count * sizeof(double), cudaMemcpyDeviceToHost);
    FILE* fp = fopen(path, "wb");
    if (fp) {
        fwrite(h, sizeof(double), count, fp);
        fclose(fp);
        fprintf(stderr, "  [debug] dumped %zu doubles to %s\n", count, path);
    }
    delete[] h;
}

// Which fields to dump (set via CUVHS_DUMP_FIELDS="0,28" env var, default: 0,28)
static std::vector<int> get_dump_fields() {
    std::vector<int> result;
    const char* env = getenv("CUVHS_DUMP_FIELDS");
    if (!env) env = "0,28";
    const char* p = env;
    while (*p) {
        char* end;
        long v = strtol(p, &end, 10);
        if (end != p) result.push_back((int)v);
        p = end;
        if (*p == ',') p++;
        else break;
    }
    return result;
}

static const int DUMP_SAMPLES = 20000;  // how many samples to dump per field

Pipeline::Pipeline(const GPUDevice& gpu_, const VideoFormat& fmt_,
                   RawReader& reader_, TBCWriter& writer_)
    : gpu(gpu_), fmt(fmt_), reader(reader_), writer(writer_) {}

Pipeline::~Pipeline() {
    free_buffers();
}

// Extra lines to read/demod beyond samples_per_field so that the TBC resampler
// has valid demod data for the last output lines.  With ref_line=19 and
// active_line_start=10, the last output line (linelocs[272]) needs data
// ~16K samples past the nominal field boundary.  NTSC vblank = 9H, so
// 20 lines (35.6K samples) covers this with margin.
static const int FIELD_MARGIN_LINES = 20;

size_t Pipeline::bytes_per_field() const {
    size_t spf_padded = fmt.samples_per_field + FIELD_MARGIN_LINES * fmt.samples_per_line;
    size_t raw = spf_padded * sizeof(double);                  // input (converted, padded)
    size_t demod = spf_padded * sizeof(double);                // demod output (padded)
    size_t demod_05 = spf_padded * sizeof(double);             // sync demod (padded)
    size_t envelope = spf_padded * sizeof(double);             // RF envelope magnitude
    size_t pulses = MAX_PULSES * 3 * sizeof(int) + sizeof(int); // starts + lengths + types + count
    size_t linelocs = fmt.lines_per_frame * sizeof(double);    // line locations
    size_t tbc = fmt.output_line_len * fmt.output_field_lines * sizeof(uint16_t);  // luma
    size_t chroma = tbc;                                       // chroma

    // K1 (FM demod) scratch buffers: FFT half, analytic, post-FFT
    size_t k1_scratch = FMDemodState::scratch_bytes_per_field((int)spf_padded);

    // K7 (dropout detection) output buffers
    size_t dropouts = MAX_DROPOUTS_PER_FIELD * 3 * sizeof(int) + sizeof(int);  // lines + starts + ends + count

    // cuFFT internal workspace (estimated per field, scales linearly with batch)
    int fft_size = (int)spf_padded;
    // Round up to 7-smooth for cuFFT (same logic as fm_demod.cu)
    while (true) {
        int n = fft_size;
        while (n % 2 == 0) n /= 2;
        while (n % 3 == 0) n /= 3;
        while (n % 5 == 0) n /= 5;
        while (n % 7 == 0) n /= 7;
        if (n == 1) break;
        fft_size++;
    }
    int freq_bins = fft_size / 2 + 1;
    size_t ws_r2c = 0, ws_c2c = 0, ws_c2r = 0;
    {
        int n[] = { fft_size };
        cufftEstimateMany(1, n, NULL, 1, fft_size, NULL, 1, freq_bins, CUFFT_D2Z, 1, &ws_r2c);
        cufftEstimateMany(1, n, NULL, 1, fft_size, NULL, 1, fft_size, CUFFT_Z2Z, 1, &ws_c2c);
        cufftEstimateMany(1, n, NULL, 1, freq_bins, NULL, 1, fft_size, CUFFT_Z2D, 1, &ws_c2r);
    }
    size_t cufft_workspace = ws_r2c + ws_c2c + ws_c2r;

    return raw + demod + demod_05 + envelope + pulses + linelocs + tbc + chroma + k1_scratch + dropouts + cufft_workspace;
}

bool Pipeline::allocate_buffers() {
    // Try-and-backoff: start aggressive (95% of free VRAM), back off on failure.
    // This maximizes batch size without needing to predict cuFFT workspace overhead.
    for (double fraction = 0.95; fraction >= 0.50; fraction -= 0.05) {
        batch_size = gpu.max_batch_size(bytes_per_field(), fraction);

        // In stream mode, use smaller batches for lower latency
        if (reader.is_stream() && batch_size > 64) batch_size = 64;
        if (batch_size < 1) batch_size = 1;

        fprintf(stderr, "Trying %.0f%% VRAM → batch %d fields (%.1f MB per field, %.1f MB total)%s\n",
                fraction * 100, batch_size,
                bytes_per_field() / (1024.0 * 1024.0),
                batch_size * bytes_per_field() / (1024.0 * 1024.0),
                reader.is_stream() ? " [stream mode]" : "");

        size_t n = batch_size;
        size_t spf = fmt.samples_per_field;
        size_t spf_padded = spf + FIELD_MARGIN_LINES * fmt.samples_per_line;
        size_t tbc_field_size = fmt.output_line_len * fmt.output_field_lines;

        size_t vram_free_before, vram_total;
        cudaMemGetInfo(&vram_free_before, &vram_total);

        bool ok = true;
        auto alloc = [&ok](void** ptr, size_t bytes) {
            if (!ok) return;
            cudaError_t err = cudaMalloc(ptr, bytes);
            if (err != cudaSuccess) {
                ok = false;
                cudaGetLastError();  // clear the error
            }
        };

        // All raw/demod buffers use spf_padded stride so each field's demod data
        // includes continuation samples past the nominal boundary (for TBC resampler).
        alloc(&d_raw,        n * spf_padded * sizeof(double));
        alloc(&d_demod,      n * spf_padded * sizeof(double));
        alloc(&d_demod_05,   n * spf_padded * sizeof(double));
        alloc(&d_envelope,   n * spf_padded * sizeof(double));
        alloc(&d_pulse_starts,  n * MAX_PULSES * sizeof(int));
        alloc(&d_pulse_lengths, n * MAX_PULSES * sizeof(int));
        alloc(&d_pulse_count,   n * sizeof(int));
        alloc(&d_pulse_types,   n * MAX_PULSES * sizeof(int));
        alloc(&d_linelocs,   n * fmt.lines_per_frame * sizeof(double));
        alloc(&d_is_first_field, n * sizeof(int));
        alloc(&d_tbc_luma,   n * tbc_field_size * sizeof(uint16_t));
        alloc(&d_tbc_chroma, n * tbc_field_size * sizeof(uint16_t));
        alloc(&d_do_lines,  n * MAX_DROPOUTS_PER_FIELD * sizeof(int));
        alloc(&d_do_starts, n * MAX_DROPOUTS_PER_FIELD * sizeof(int));
        alloc(&d_do_ends,   n * MAX_DROPOUTS_PER_FIELD * sizeof(int));
        alloc(&d_do_count,  n * sizeof(int));

        // Initialize FM demod state (cuFFT plans + filter arrays + scratch buffers)
        // Use spf_padded so FFT plans are sized for the larger field
        if (ok) ok = demod_state.init(fmt, batch_size, (int)spf_padded);

        if (ok) {
            fprintf(stderr, "  Field margin: %d lines (%zu extra samples, spf_padded=%zu)\n",
                    FIELD_MARGIN_LINES, spf_padded - spf, spf_padded);
            size_t vram_free_after, vram_dummy;
            cudaMemGetInfo(&vram_free_after, &vram_dummy);
            fprintf(stderr, "  VRAM used: %.1f GB, remaining: %.1f GB\n",
                    (vram_free_before - vram_free_after) / 1e9, vram_free_after / 1e9);
            return true;
        }

        // Allocation failed at this fraction — clean up and try a smaller batch
        fprintf(stderr, "  Allocation failed at %.0f%%, backing off...\n", fraction * 100);
        demod_state.destroy();
        free_buffers();
    }

    fprintf(stderr, "Failed to allocate GPU buffers even at 50%% VRAM\n");
    return false;
}

void Pipeline::free_buffers() {
    auto safe_free = [](void** ptr) {
        if (*ptr) { cudaFree(*ptr); *ptr = nullptr; }
    };
    safe_free(&d_raw);
    safe_free(&d_demod);
    safe_free(&d_demod_05);
    safe_free(&d_envelope);
    safe_free(&d_pulse_starts);
    safe_free(&d_pulse_lengths);
    safe_free(&d_pulse_count);
    safe_free(&d_pulse_types);
    safe_free(&d_linelocs);
    safe_free(&d_is_first_field);
    safe_free(&d_tbc_luma);
    safe_free(&d_tbc_chroma);
    safe_free(&d_do_lines);
    safe_free(&d_do_starts);
    safe_free(&d_do_ends);
    safe_free(&d_do_count);
}

// ============================================================================
// Pre-scan: find VSYNC-based field boundaries in the raw file
// ============================================================================
//
// CPU envelope scan detects one field type (type A) reliably from the raw u8
// amplitude. The other type (B) has a subtler VSYNC that doesn't cross the
// amplitude threshold. We interpolate type-B positions as midpoints between
// consecutive type-A fields, then merge and sort.
//
// K3's VSYNC state machine handles ±500 sample position errors naturally,
// so interpolated positions don't need to be exact.

bool Pipeline::prescan_field_boundaries() {
    if (reader.is_stream()) {
        return true;  // stream mode: field boundaries found inline
    }

    size_t total_samples = reader.total_samples();
    size_t spf = fmt.samples_per_field;

    fprintf(stderr, "Pre-scanning for field boundaries...\n");
    auto t0 = std::chrono::steady_clock::now();

    // Dual sliding window parameters
    int short_win = (int)(fmt.samples_per_line / 2);   // ~890 samples (half a line)
    if (short_win < 100) short_win = 100;
    size_t long_win = spf * 50;                         // ~50 fields for local baseline
    int vsync_min_run = (int)(fmt.vsync_width * 0.5);

    // Chunked I/O: 256 MB chunks, overlapping by long_win
    size_t chunk_bytes = 256 * 1024 * 1024;
    size_t overlap = long_win;
    auto* buf = new uint8_t[chunk_bytes + overlap];

    std::vector<size_t> type_a;
    size_t last_end = 0;

    // Sliding window state (carried across chunks)
    double short_sum = 0.0;
    double long_sum = 0.0;
    size_t long_count = 0;    // grows from short_win up to long_win
    int run_len = 0;

    int num_chunks = (int)((total_samples + chunk_bytes - 1) / chunk_bytes);
    int chunk_idx = 0;

    for (size_t chunk_start = 0; chunk_start < total_samples; ) {
        // Read this chunk (with overlap from previous if not first)
        size_t read_start = (chunk_start > overlap && chunk_idx > 0)
                            ? chunk_start - overlap : chunk_start;
        size_t read_len = std::min(chunk_bytes + overlap, total_samples - read_start);
        size_t n_read = reader.read_raw_at(buf, read_start, read_len);
        if (n_read < (size_t)short_win) break;

        // On first chunk, initialize short window
        size_t scan_start_in_buf = (chunk_idx == 0) ? (size_t)short_win : overlap;

        if (chunk_idx == 0) {
            short_sum = 0.0;
            for (int i = 0; i < short_win; i++) short_sum += buf[i];
            long_sum = short_sum;
            long_count = short_win;
        } else {
            // Re-initialize windows from overlap region
            short_sum = 0.0;
            for (size_t i = scan_start_in_buf - short_win; i < scan_start_in_buf; i++)
                short_sum += buf[i];
            long_sum = 0.0;
            size_t lw_actual = std::min(long_count, scan_start_in_buf);
            for (size_t i = scan_start_in_buf - lw_actual; i < scan_start_in_buf; i++)
                long_sum += buf[i];
            long_count = lw_actual;
        }

        // Scan this chunk
        for (size_t i = scan_start_in_buf; i < n_read; i++) {
            // Update short window
            short_sum += buf[i];
            short_sum -= buf[i - short_win];

            // Update long window (growing phase: expand until long_win reached)
            long_sum += buf[i];
            if (long_count < long_win) {
                long_count++;
            } else {
                long_sum -= buf[i - long_win];
            }

            double short_mean = short_sum / short_win;
            double long_mean = long_sum / (double)long_count;

            size_t abs_pos = read_start + i;

            if (short_mean < long_mean * 0.85) {
                run_len++;
            } else {
                if (run_len >= vsync_min_run) {
                    if (type_a.empty() || (abs_pos - last_end) > spf / 2) {
                        type_a.push_back(abs_pos);
                        last_end = abs_pos;
                    }
                }
                run_len = 0;
            }
        }

        chunk_start += chunk_bytes;
        chunk_idx++;
        fprintf(stderr, "\r  [prescan] chunk %d/%d (%.0f%%)...",
                chunk_idx, num_chunks, 100.0 * chunk_start / total_samples);
    }
    fprintf(stderr, "\n");

    delete[] buf;

    auto t_scan = std::chrono::steady_clock::now();
    double scan_ms = std::chrono::duration<double, std::milli>(t_scan - t0).count();

    fprintf(stderr, "  [prescan] envelope scan: %.0fms, found %zu type-A fields\n",
            scan_ms, type_a.size());

    if (type_a.size() < 2) {
        fprintf(stderr, "  Not enough type-A fields found\n");
        return false;
    }

    // Phase 2: build full field offset list
    //
    // Type-A detections give us every-other VSYNC (~955K apart = 2 fields).
    // The missing type-B fields sit at the midpoints between consecutive
    // type-A fields. K3 handles ±500 sample position errors naturally.
    //
    // Special case: field 0 starts at offset 0 (before the first VSYNC),
    // matching Python. The first type-A detection IS field 1's VSYNC.
    // We only interpolate between consecutive type-A fields after that.
    field_offsets.clear();
    // No fixed backtrack — the VSYNC dip end position is used directly.
    // K3 searches the full spf-sized chunk for the VSYNC state machine,
    // so it doesn't matter exactly where the VSYNC falls within the chunk.
    // This avoids a tape/VCR-specific calibration constant.
    size_t backtrack = 0;

    // Field 0: starts at sample 0
    field_offsets.push_back(0);

    // Field 1: first type-A VSYNC position
    if (!type_a.empty()) {
        size_t a_start = (type_a[0] > backtrack) ? type_a[0] - backtrack : 0;
        field_offsets.push_back(a_start);
    }

    // Fields 2+: for each consecutive pair of type-A VSYNCs,
    // insert a type-B midpoint, then the next type-A
    for (size_t i = 1; i < type_a.size(); i++) {
        // Type-B: midpoint between type_a[i-1] and type_a[i]
        size_t midpoint = (type_a[i - 1] + type_a[i]) / 2;
        size_t b_start = (midpoint > backtrack) ? midpoint - backtrack : 0;
        field_offsets.push_back(b_start);

        // Type-A: this VSYNC
        size_t a_start = (type_a[i] > backtrack) ? type_a[i] - backtrack : 0;
        field_offsets.push_back(a_start);
    }

    auto t_done = std::chrono::steady_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t_done - t0).count();

    fprintf(stderr, "  [prescan] %.0fms total, %zu fields (%zu type-A + %zu interpolated)\n",
            total_ms, field_offsets.size(), type_a.size(),
            field_offsets.size() - type_a.size());
    if (field_offsets.size() >= 2) {
        double avg_spacing = (double)(field_offsets.back() - field_offsets.front())
                             / (field_offsets.size() - 1);
        fprintf(stderr, "  Average field spacing: %.0f samples (nominal: %zu)\n",
                avg_spacing, spf);
    }

    // Dump field offsets for comparison with Python
    {
        FILE* fp = fopen("/tmp/cuvhs_field_offsets.txt", "w");
        if (fp) {
            for (size_t i = 0; i < field_offsets.size(); i++) {
                fprintf(fp, "field %zu: offset %zu\n", i, field_offsets[i]);
            }
            fclose(fp);
            fprintf(stderr, "  [debug] wrote field offsets to /tmp/cuvhs_field_offsets.txt\n");
        }
    }

    return !field_offsets.empty();
}

// ============================================================================
// Process a batch of fields using pre-scanned offsets
// ============================================================================

int Pipeline::process_batch(int start_field, int num_fields) {
    size_t spf = fmt.samples_per_field;
    size_t spf_padded = spf + FIELD_MARGIN_LINES * fmt.samples_per_line;
    size_t tbc_field_size = fmt.output_line_len * fmt.output_field_lines;

    // Read each field from its pre-scanned offset (spf_padded samples each)
    auto* h_raw = new double[num_fields * spf_padded];
    int fields_loaded = 0;

    for (int i = 0; i < num_fields; i++) {
        int fi = start_field + i;
        if (fi >= (int)field_offsets.size()) break;

        size_t offset = field_offsets[fi];
        size_t n_read = reader.read_at(h_raw + (size_t)i * spf_padded, offset, spf_padded);
        if (n_read < spf / 2) break;
        // Zero-pad if short
        if (n_read < spf_padded) {
            memset(h_raw + (size_t)i * spf_padded + n_read, 0, (spf_padded - n_read) * sizeof(double));
        }
        fields_loaded++;
    }

    if (fields_loaded == 0) {
        delete[] h_raw;
        return 0;
    }

    // 2. Upload to GPU
    cudaMemcpy(d_raw, h_raw, fields_loaded * spf_padded * sizeof(double),
               cudaMemcpyHostToDevice);
    delete[] h_raw;

    // --- Debug: dump pre-K1 raw data for target fields ---
    {
        static auto dump_fields = get_dump_fields();
        for (int df : dump_fields) {
            int local_idx = df - start_field;
            if (local_idx >= 0 && local_idx < fields_loaded) {
                char path[256];
                snprintf(path, sizeof(path), "/tmp/cuvhs_raw_field%d.bin", df);
                size_t dump_n = std::min((size_t)DUMP_SAMPLES, spf_padded);
                dump_gpu_doubles(path,
                    static_cast<double*>(d_raw) + (size_t)local_idx * spf_padded,
                    dump_n);
            }
        }
    }

    // 3. FM Demodulation (Kernel 1) — demod spf_padded samples per field
    fm_demod(demod_state,
             static_cast<double*>(d_raw),
             static_cast<double*>(d_demod),
             static_cast<double*>(d_demod_05),
             static_cast<double*>(d_envelope),
             fields_loaded, spf_padded, fmt);

    // --- Debug: dump post-K1 demod_05 for target fields ---
    {
        static auto dump_fields = get_dump_fields();
        for (int df : dump_fields) {
            int local_idx = df - start_field;
            if (local_idx >= 0 && local_idx < fields_loaded) {
                char path[256];
                snprintf(path, sizeof(path), "/tmp/cuvhs_demod05_field%d.bin", df);
                size_t dump_n = std::min((size_t)DUMP_SAMPLES, spf_padded);
                dump_gpu_doubles(path,
                    static_cast<double*>(d_demod_05) + (size_t)local_idx * spf_padded,
                    dump_n);
            }
        }
    }

    // 4. Find Sync Pulses (Kernel 2) — uses spf_padded stride for buffer addressing
    sync_pulses(static_cast<double*>(d_demod_05),
                static_cast<int*>(d_pulse_starts),
                static_cast<int*>(d_pulse_lengths),
                static_cast<int*>(d_pulse_count),
                fields_loaded, spf_padded, fmt);

    // --- Debug: dump pulse counts ---
    {
        auto* h_pc = new int[fields_loaded];
        cudaMemcpy(h_pc, d_pulse_count, fields_loaded * sizeof(int), cudaMemcpyDeviceToHost);
        static auto dump_fields = get_dump_fields();
        for (int df : dump_fields) {
            int local_idx = df - start_field;
            if (local_idx >= 0 && local_idx < fields_loaded) {
                fprintf(stderr, "  [debug] field %d: %d pulses\n", df, h_pc[local_idx]);
            }
        }
        delete[] h_pc;
    }

    // 5. Classify Pulses + Line Locations (Kernel 3)
    line_locs(static_cast<int*>(d_pulse_starts),
              static_cast<int*>(d_pulse_lengths),
              static_cast<int*>(d_pulse_count),
              static_cast<int*>(d_pulse_types),
              static_cast<double*>(d_linelocs),
              static_cast<int*>(d_is_first_field),
              fields_loaded, fmt);

    // 6. Refine Line Locations via Hsync Zero-Crossing (Kernel 4)
    hsync_refine(static_cast<double*>(d_demod_05),
                 static_cast<double*>(d_linelocs),
                 fields_loaded,
                 fields_loaded * (int)spf_padded,
                 fmt);

    // --- Debug: dump linelocs for target fields ---
    {
        static auto dump_fields = get_dump_fields();
        for (int df : dump_fields) {
            int local_idx = df - start_field;
            if (local_idx >= 0 && local_idx < fields_loaded) {
                char path[256];
                snprintf(path, sizeof(path), "/tmp/cuvhs_linelocs_field%d.bin", df);
                dump_gpu_doubles(path,
                    static_cast<double*>(d_linelocs) + (size_t)local_idx * fmt.lines_per_frame,
                    fmt.lines_per_frame);
            }
        }
    }

    // 7. TBC Resample (Kernel 5) — total_demod_samples = fields * spf_padded
    tbc_resample(static_cast<double*>(d_demod),
                 static_cast<double*>(d_linelocs),
                 static_cast<uint16_t*>(d_tbc_luma),
                 fields_loaded,
                 fields_loaded * (int)spf_padded,
                 fmt);

    // 8. Chroma Decode (Kernel 6)
    // d_demod is reused as scratch — K5 (luma resample) is done with it
    // Pass &chroma_state to carry track/phase info across batch boundaries
    std::vector<int> field_phase_ids;
    chroma_decode(static_cast<double*>(d_raw),
                  static_cast<double*>(d_linelocs),
                  static_cast<double*>(d_demod),
                  static_cast<uint16_t*>(d_tbc_chroma),
                  fields_loaded,
                  fields_loaded * (int)spf_padded,
                  fmt,
                  field_phase_ids,
                  &chroma_state);

    // 9. Dropout Detection + Concealment (Kernel 7)
    dropout_detect(static_cast<double*>(d_envelope),
                   static_cast<double*>(d_linelocs),
                   static_cast<uint16_t*>(d_tbc_luma),
                   static_cast<uint16_t*>(d_tbc_chroma),
                   static_cast<int*>(d_do_lines),
                   static_cast<int*>(d_do_starts),
                   static_cast<int*>(d_do_ends),
                   static_cast<int*>(d_do_count),
                   fields_loaded, spf_padded, fmt);

    // 10. Download TBC results + dropout metadata and write to disk
    auto* h_luma = new uint16_t[fields_loaded * tbc_field_size];
    auto* h_chroma = new uint16_t[fields_loaded * tbc_field_size];

    size_t do_buf_size = (size_t)fields_loaded * MAX_DROPOUTS_PER_FIELD;
    auto* h_do_lines  = new int[do_buf_size];
    auto* h_do_starts = new int[do_buf_size];
    auto* h_do_ends   = new int[do_buf_size];
    auto* h_do_count  = new int[fields_loaded];

    cudaMemcpy(h_luma, d_tbc_luma,
               fields_loaded * tbc_field_size * sizeof(uint16_t),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_chroma, d_tbc_chroma,
               fields_loaded * tbc_field_size * sizeof(uint16_t),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(h_do_lines, d_do_lines, do_buf_size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_do_starts, d_do_starts, do_buf_size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_do_ends, d_do_ends, do_buf_size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_do_count, d_do_count, fields_loaded * sizeof(int), cudaMemcpyDeviceToHost);

    // Download field parity
    auto* h_is_first = new int[fields_loaded];
    cudaMemcpy(h_is_first, d_is_first_field, fields_loaded * sizeof(int), cudaMemcpyDeviceToHost);

    static bool last_parity = true;

    for (int i = 0; i < fields_loaded; i++) {
        writer.write_luma_field(h_luma + i * tbc_field_size);
        writer.write_chroma_field(h_chroma + i * tbc_field_size);

        // Field parity: use VSYNC-based detection, but enforce alternation.
        // Two consecutive fields can never have the same parity in valid interlaced video.
        // Detection is only useful to establish initial parity or correct after a glitch.
        bool first_field;
        if (h_is_first[i] >= 0 && (h_is_first[i] == 1) != last_parity) {
            // Detection agrees with alternation — trust it
            first_field = (h_is_first[i] == 1);
        } else {
            // Unknown OR detection conflicts with alternation — alternate
            first_field = !last_parity;
        }
        last_parity = first_field;
        writer.set_first_field(first_field);

        // Set NTSC field phase ID if available
        if (i < (int)field_phase_ids.size() && field_phase_ids[i] > 0) {
            writer.set_field_phase_id(field_phase_ids[i]);
        }

        // Record dropout metadata for JSON output
        int n_do = h_do_count[i];
        int base = i * MAX_DROPOUTS_PER_FIELD;
        for (int d = 0; d < n_do; d++) {
            writer.add_dropout(h_do_lines[base + d],
                               h_do_starts[base + d],
                               h_do_ends[base + d]);
        }

        writer.finish_field();
    }

    delete[] h_luma;
    delete[] h_chroma;
    delete[] h_do_lines;
    delete[] h_do_starts;
    delete[] h_do_ends;
    delete[] h_do_count;
    delete[] h_is_first;

    return fields_loaded;
}

bool Pipeline::run() {
    if (!allocate_buffers())
        return false;

    bool streaming = reader.is_stream();

    // Pre-scan to find field boundaries (file mode only)
    if (!streaming) {
        if (!prescan_field_boundaries()) {
            fprintf(stderr, "Failed to find any field boundaries\n");
            return false;
        }
    }

    int total_fields_est = streaming ? 0 : (int)field_offsets.size();

    if (streaming) {
        fprintf(stderr, "Starting decode: streaming mode, batches of %d fields\n", batch_size);
    } else {
        fprintf(stderr, "Starting decode: %d fields in batches of %d\n",
                total_fields_est, batch_size);
    }

    auto t_start = std::chrono::steady_clock::now();
    int total_fields = 0;

    // Progress display: print two lines initially so \033[A can move up
    fprintf(stderr, "\n\n");
    fflush(stderr);

    while (true) {
        int fields_this_batch = batch_size;

        if (!streaming) {
            int remaining = total_fields_est - total_fields;
            if (remaining <= 0) break;
            fields_this_batch = std::min(batch_size, remaining);
        }

        int processed = process_batch(total_fields, fields_this_batch);
        if (processed == 0) break;  // EOF or error

        total_fields += processed;

        // Write JSON after each batch so partial results are usable if we crash/get killed
        writer.write_json();

        // Progress dashboard (two lines, rewritten in-place)
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - t_start).count();
        double fps = (total_fields / 2.0) / elapsed;

        // Move cursor up 2 lines, clear them
        fprintf(stderr, "\033[2A\033[K");

        if (streaming) {
            fprintf(stderr, "  %d fields | %.1f FPS | %.0fs elapsed (streaming)\n\033[K\n",
                    total_fields, fps, elapsed);
        } else {
            double pct = 100.0 * total_fields / total_fields_est;
            double eta = (pct > 0.1) ? elapsed * (100.0 - pct) / pct : 0.0;

            // Progress bar (40 chars wide)
            int bar_fill = (int)(pct * 40.0 / 100.0);
            if (bar_fill > 40) bar_fill = 40;
            char bar[42];
            for (int i = 0; i < 40; i++) bar[i] = (i < bar_fill) ? '#' : '-';
            bar[40] = '\0';

            int eta_min = (int)(eta / 60.0);
            int eta_sec = (int)(eta) % 60;

            fprintf(stderr, "  %d/%d fields (%.1f%%) | %.1f FPS | ETA %d:%02d\n",
                    total_fields, total_fields_est, pct, fps, eta_min, eta_sec);
            fprintf(stderr, "\033[K  [%s]\n", bar);
        }
        fflush(stderr);
    }

    auto t_end = std::chrono::steady_clock::now();
    double total_time = std::chrono::duration<double>(t_end - t_start).count();

    // Final summary (overwrite the progress lines)
    fprintf(stderr, "\033[2A\033[K");
    fprintf(stderr, "Decode complete: %d fields in %.1f seconds (%.1f FPS)\n\033[K\n",
            total_fields, total_time, (total_fields / 2.0) / total_time);
    fflush(stderr);

    writer.finalize();
    return true;
}
