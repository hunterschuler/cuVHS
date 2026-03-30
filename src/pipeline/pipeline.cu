#include "pipeline/pipeline.h"
#include "pipeline/fm_demod.h"
#include "pipeline/sync_pulses.h"
#include "pipeline/line_locs.h"
#include "pipeline/hsync_refine.h"
#include "pipeline/tbc_resample.h"
#include "pipeline/chroma_decode.h"
#include "pipeline/dropout_detect.h"
#include "pipeline/vsync_discover.h"

#include <cuda_runtime.h>
#include <cufft.h>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <algorithm>

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
        // Allocate space for up to 20 candidate pulses per field (very generous)
        alloc(&d_candidate_indices, n * 20 * sizeof(int));
        // Just one integer for the global atomic counter
        alloc(&d_candidate_count, sizeof(int));        
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
    safe_free(&d_candidate_indices);
    safe_free(&d_candidate_count);
}

// ============================================================================
// Process one chunk: demod, find VSYNC inline, process through full pipeline
// ============================================================================
//
// Reads a contiguous chunk of raw samples, demodulates it, finds VSYNC positions
// in the demod domain using K2/K3, then processes the fields through the full
// pipeline (K4-K7). All in one pass - no separate prescan needed.
//
// NTSC note: Assumes ~955K samples per field (29.97 fps, 31336 samples/line, 262.5 lines).
// PAL would need different nominal spacing.

int Pipeline::process_chunk(size_t raw_offset, int num_fields, size_t& next_raw_offset) {
    size_t spf = fmt.samples_per_field;
    size_t spf_padded = spf + FIELD_MARGIN_LINES * fmt.samples_per_line;
    size_t tbc_field_size = fmt.output_line_len * fmt.output_field_lines;

    // Read contiguous raw: num_fields * spf_padded (each field needs spf_padded samples)
    size_t total_samples = (size_t)num_fields * spf_padded;
    auto* h_raw = new double[total_samples];

    size_t n_read = reader.read_at(h_raw, raw_offset, total_samples);
    if (n_read < spf) {
        delete[] h_raw;
        return 0;
    }
    if (n_read < total_samples) {
        memset(h_raw + n_read, 0, (total_samples - n_read) * sizeof(double));
    }

    int fields_loaded = (int)(n_read / spf);
    if (fields_loaded > num_fields) fields_loaded = num_fields;
    if (fields_loaded == 0) {
        delete[] h_raw;
        return 0;
    }

    // Upload contiguous raw to GPU
    cudaMemcpy(d_raw, h_raw, total_samples * sizeof(double), cudaMemcpyHostToDevice);
    delete[] h_raw;

    // ================================================================
    // K1: FM demod → contiguous d_demod, d_demod_05, d_envelope
    // ================================================================
    fm_demod(demod_state,
             static_cast<double*>(d_raw),
             static_cast<double*>(d_demod),
             static_cast<double*>(d_demod_05),
             static_cast<double*>(d_envelope),
             fields_loaded, spf_padded, fmt);

    // ================================================================
    // NEW: Global Pulse Discovery (Replaces uniform-stride K2/K3)
    // ================================================================
    int total_chunk_samples = fields_loaded * spf_padded;
    int candidate_capacity = fields_loaded * 20;
    
    // Ensure the counter is zeroed out before launching
    cudaMemset(d_candidate_count, 0, sizeof(int));

    discover_vsyncs(static_cast<double*>(d_demod_05),
                    static_cast<int*>(d_candidate_indices),
                    static_cast<int*>(d_candidate_count),
                    candidate_capacity,
                    total_chunk_samples,
                    fmt);

    // Download candidates to CPU
    int num_candidates;
    cudaMemcpy(&num_candidates, d_candidate_count, sizeof(int), cudaMemcpyDeviceToHost);
    int stored_candidates = std::min(num_candidates, candidate_capacity);
    if (num_candidates > candidate_capacity) {
        fprintf(stderr, "Warning: VSYNC candidate buffer overflow (%d > %d), truncating chunk results\n",
                num_candidates, candidate_capacity);
    }

    std::vector<int> h_candidates(stored_candidates);
    if (stored_candidates > 0) {
        cudaMemcpy(h_candidates.data(), d_candidate_indices,
                   (size_t)stored_candidates * sizeof(int), cudaMemcpyDeviceToHost);
    }

    // Atomic adds don't guarantee order, so sort the indices
    std::sort(h_candidates.begin(), h_candidates.end());

    // Cluster the 27µs pulses into Field boundaries
    std::vector<size_t> chunk_field_offsets;
    for (int pos : h_candidates) {
        // If this is the first pulse, or it's at least half a field away from the last recorded boundary
        if (chunk_field_offsets.empty() || 
           (pos - chunk_field_offsets.back() > fmt.samples_per_field / 2)) 
        {
            // Back up ~5 lines so the K2/K3 window starts BEFORE the VSYNC.
            // This guarantees K3's state machine sees the required leading HSYNCs.
            size_t safe_offset = (pos > (5 * fmt.samples_per_line)) ? 
                                 (pos - 5 * fmt.samples_per_line) : 0;
            
            chunk_field_offsets.push_back(safe_offset);
        }
    }

    if (chunk_field_offsets.empty()) {
        fprintf(stderr, "Warning: no VSYNC candidates found in chunk at raw offset %zu, using nominal field spacing fallback\n",
                raw_offset);
        for (int i = 0; i < fields_loaded; i++) {
            chunk_field_offsets.push_back((size_t)i * spf);
        }
    }

    fields_loaded = (int)chunk_field_offsets.size();
    if (fields_loaded > num_fields) {
        fields_loaded = num_fields;  // cap to buffer allocation
        chunk_field_offsets.resize(fields_loaded);
    }

    // Compute actual field spacing from detected fields (handles VCR speed drift)
    size_t actual_spf = spf;  // default to nominal
    if (chunk_field_offsets.size() >= 2) {
        size_t total_span = chunk_field_offsets.back() - chunk_field_offsets.front();
        size_t num_gaps = chunk_field_offsets.size() - 1;
        actual_spf = total_span / num_gaps;
    }

    // ----------------------------------------------------------------
    // NEW: The "Look-Back" Stitching Margin
    // ----------------------------------------------------------------
    // Predict where the NEXT field's VSYNC will start
    size_t predicted_next_field = chunk_field_offsets.back() + actual_spf;

    // Pull the start pointer backwards by 10 horizontal lines. 
    // This creates a microscopic overlap (~30,000 samples) ensuring the 
    // next GPU batch swallows the entire VSYNC block whole, even if the tape sped up.
    size_t safe_margin = 10 * fmt.samples_per_line;

    if (predicted_next_field > safe_margin) {
        next_raw_offset = raw_offset + predicted_next_field - safe_margin;
    } else {
        next_raw_offset = raw_offset + predicted_next_field; // Fallback
    }

    // --- REMOVED delete[] h_k3_debug; ---

    // Debug output to file for comparison with vhs-decode
    static FILE* debug_fp = nullptr;
    if (debug_fp == nullptr) {
        debug_fp = fopen("/tmp/cuvhs_debug.txt", "w");
    }
    if (debug_fp) {
        fprintf(debug_fp, "Chunk at raw_offset=%zu (%.1f sec):\n",
                raw_offset, (double)raw_offset / 28000000.0);
        for (size_t i = 0; i < chunk_field_offsets.size(); i++) {
            fprintf(debug_fp, "  Field %zu: chunk_offset=%zu file_offset=%zu (%.3f sec)\n",
                    i, chunk_field_offsets[i], raw_offset + chunk_field_offsets[i],
                    (double)(raw_offset + chunk_field_offsets[i]) / 28000000.0);
        }
        fprintf(debug_fp, "  actual_spf=%zu (nominal=%zu, diff=%+.0f)\n",
                actual_spf, spf, (double)actual_spf - (double)spf);
        fprintf(debug_fp, "  next_raw_offset=%zu\n", next_raw_offset);
        
        // --- FIXED: Replaced unique_vsyncs with num_candidates ---
        fprintf(debug_fp, "  [prescan] %d candidate pulses (%d stored) → %d fields\n\n",
                num_candidates, stored_candidates, fields_loaded);
        fflush(debug_fp);
    }

    // ================================================================
    // Upload field offsets to GPU and re-run K2 with correct offsets
    // ================================================================
    size_t* d_field_offsets = nullptr;
    cudaMalloc(&d_field_offsets, fields_loaded * sizeof(size_t));
    cudaMemcpy(d_field_offsets, chunk_field_offsets.data(),
               fields_loaded * sizeof(size_t), cudaMemcpyHostToDevice);

    // K2 (re-run): find pulses at correct field boundaries
    sync_pulses(static_cast<double*>(d_demod_05),
                static_cast<int*>(d_pulse_starts),
                static_cast<int*>(d_pulse_lengths),
                static_cast<int*>(d_pulse_count),
                d_field_offsets,
                fields_loaded, spf_padded, fmt);

    // K3: Classify pulses + line locations (now with correct field alignment)
    K3Debug* d_k3_debug = nullptr;
    {
        static auto dump_fields = get_dump_fields();
        if (!dump_fields.empty()) {
            cudaMalloc(&d_k3_debug, fields_loaded * sizeof(K3Debug));
            cudaMemset(d_k3_debug, 0, fields_loaded * sizeof(K3Debug));
        }
    }
    line_locs(static_cast<int*>(d_pulse_starts),
              static_cast<int*>(d_pulse_lengths),
              static_cast<int*>(d_pulse_count),
              static_cast<int*>(d_pulse_types),
              static_cast<double*>(d_linelocs),
              static_cast<int*>(d_is_first_field),
              fields_loaded, fmt, d_k3_debug);

    // Debug: dump K3 decisions
    if (d_k3_debug) {
        static auto dump_fields = get_dump_fields();
        auto* h_dbg = new K3Debug[fields_loaded];
        cudaMemcpy(h_dbg, d_k3_debug, fields_loaded * sizeof(K3Debug), cudaMemcpyDeviceToHost);
        for (int df : dump_fields) {
            int local_idx = df;
            if (local_idx >= 0 && local_idx < fields_loaded) {
                const K3Debug& d = h_dbg[local_idx];
                fprintf(stderr, "  [K3 debug] field %d: npc=%d ref_idx=%d ref_pos=%.1f ref_line=%.1f "
                        "meanll=%.2f best_run=%d num_hsyncs=%d hsync_off=%.2f state=%d parity=%d\n",
                        df, d.npc, d.ref_pulse_idx, d.ref_position, d.ref_line,
                        d.meanlinelen, d.best_run_len, d.num_hsyncs, d.hsync_offset,
                        d.final_state, d.field_parity);
            }
        }
        delete[] h_dbg;
        cudaFree(d_k3_debug);
    }

    // K4: Refine Line Locations via Hsync Zero-Crossing
    hsync_refine(static_cast<double*>(d_demod_05),
                 static_cast<double*>(d_linelocs),
                 fields_loaded,
                 fields_loaded * (int)spf_padded,
                 fmt);

    // K5: TBC Resample
    tbc_resample(static_cast<double*>(d_demod),
                 static_cast<double*>(d_linelocs),
                 static_cast<uint16_t*>(d_tbc_luma),
                 fields_loaded,
                 fields_loaded * (int)spf_padded,
                 fmt);

    // K6: Chroma Decode
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

    // K7: Dropout Detection + Concealment
    dropout_detect(static_cast<double*>(d_envelope),
                   static_cast<double*>(d_linelocs),
                   static_cast<uint16_t*>(d_tbc_luma),
                   static_cast<uint16_t*>(d_tbc_chroma),
                   static_cast<int*>(d_do_lines),
                   static_cast<int*>(d_do_starts),
                   static_cast<int*>(d_do_ends),
                   static_cast<int*>(d_do_count),
                   fields_loaded, spf_padded, fmt);

    // Download TBC results + dropout metadata and write to disk
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
        // Set fileLoc for JSON output (for debugging field position alignment)
        size_t file_loc = raw_offset + chunk_field_offsets[(size_t)i];
        writer.set_file_loc(file_loc);
        
        writer.write_luma_field(h_luma + i * tbc_field_size);
        writer.write_chroma_field(h_chroma + i * tbc_field_size);

        // Field parity: use VSYNC-based detection, but enforce alternation.
        bool first_field;
        if (h_is_first[i] >= 0 && (h_is_first[i] == 1) != last_parity) {
            first_field = (h_is_first[i] == 1);
        } else {
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
    cudaFree(d_field_offsets);

    return fields_loaded;
}

bool Pipeline::run() {
    if (!allocate_buffers())
        return false;

    bool streaming = reader.is_stream();
    size_t total_samples = streaming ? 0 : reader.total_samples();

    int total_fields_est = streaming ? 0 : (int)(total_samples / fmt.samples_per_field);

    if (streaming) {
        fprintf(stderr, "Starting decode: streaming mode, chunks of %d fields\n", batch_size);
    } else {
        fprintf(stderr, "Starting decode: ~%d fields in chunks of %d\n",
                total_fields_est, batch_size);
    }

    auto t_start = std::chrono::steady_clock::now();
    int total_fields = 0;
    size_t raw_offset = 0;

    // Progress display: print two lines initially so \033[A can move up
    fprintf(stderr, "\n\n");
    fflush(stderr);

    while (true) {
        int fields_this_chunk = batch_size;

        if (!streaming) {
            size_t remaining_samples = total_samples - raw_offset;
            int remaining_fields = (int)(remaining_samples / fmt.samples_per_field);
            if (remaining_fields <= 0) break;
            fields_this_chunk = std::min(batch_size, remaining_fields);
        }

        size_t next_raw_offset;
        int processed = process_chunk(raw_offset, fields_this_chunk, next_raw_offset);
        if (processed == 0) break;  // EOF or error

        raw_offset = next_raw_offset;
        total_fields += processed;

        // Write JSON after each chunk so partial results are usable if we crash/get killed
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
