#pragma once
#include <vector>
#include "gpu/device.h"
#include "format/video_format.h"
#include "io/raw_reader.h"
#include "io/tbc_writer.h"
#include "pipeline/fm_demod.h"
#include "pipeline/sync_pulses.h"
#include "pipeline/chroma_decode.h"
#include "pipeline/dropout_detect.h"

// Orchestrates the full decode pipeline on GPU.
//
// Data flow (all inter-kernel data stays in VRAM):
//
//   Raw samples (host) ──bulk upload──▶ GPU VRAM
//     │
//     ├─▶ Kernel 1: FM Demodulation     → demod + demod_05 (sync)
//     ├─▶ Kernel 2: Find Sync Pulses    → pulse arrays
//     ├─▶ Kernel 3: Classify + Linelocs → coarse line locations
//     ├─▶ Kernel 4: Hsync Refinement    → refined line locations
//     ├─▶ Kernel 5: TBC Resample        → luma TBC fields
//     ├─▶ Kernel 6: Chroma Decode       → chroma TBC fields
//     └─▶ Kernel 7: Dropout Detection   → dropout mask + corrected TBC
//     │
//     ◀──bulk download── TBC fields (host) → disk
//
struct Pipeline {
    Pipeline(const GPUDevice& gpu, const VideoFormat& fmt,
             RawReader& reader, TBCWriter& writer);
    ~Pipeline();

    bool run();

private:
    const GPUDevice& gpu;
    const VideoFormat& fmt;
    RawReader& reader;
    TBCWriter& writer;

    // Batch sizing determined at runtime from available VRAM
    int batch_size = 0;

    // Kernel 1: FM demodulation (persistent state — cuFFT plans + filter arrays)
    FMDemodState demod_state;

    // Kernel 6: Chroma state carried across batches (track, phase cycle)
    ChromaState chroma_state;

    // GPU buffer pointers (allocated in run(), freed in destructor)
    // These are device pointers.
    void* d_raw = nullptr;         // raw input samples (batch)
    void* d_demod = nullptr;       // FM demod output (float64, batch)
    void* d_demod_05 = nullptr;    // sync demod output (float64, batch)
    void* d_pulse_starts = nullptr;  // pulse start positions (batch x MAX_PULSES ints)
    void* d_pulse_lengths = nullptr; // pulse lengths (batch x MAX_PULSES ints)
    void* d_pulse_count = nullptr;   // pulse count per field (batch ints)
    void* d_pulse_types = nullptr;   // pulse type classification (batch x MAX_PULSES ints)
    
    // NEW: Pulse Discovery buffers
    void* d_candidate_indices = nullptr; // VSYNC candidate pulse indices (batch x 20 ints)
    void* d_candidate_count = nullptr;   // Global candidate counter (1 int)
    
    void* d_linelocs = nullptr;    // line locations (batch)
    void* d_linelocs_coarse = nullptr;  // K4 debug: pre-refine line locations
    void* d_envelope = nullptr;    // RF envelope magnitude (float64, batch)
    void* d_tbc_luma = nullptr;    // TBC luma output (uint16, batch)
    void* d_tbc_chroma = nullptr;  // TBC chroma output (uint16, batch)
    void* d_is_first_field = nullptr;  // field parity from VSYNC pulse pattern (int, batch)
    void* d_k3_bad_spacing_count = nullptr;      // K3 debug: line spacings outside tolerance
    void* d_k3_isolated_spacing_count = nullptr; // K3 debug: spacing differs from neighbors
    void* d_k3_large_jump_count = nullptr;       // K3 debug: adjacent lineloc jump > threshold
    void* d_k4_large_delta_count = nullptr;    // K4 debug: lines moved by > threshold
    void* d_k4_isolated_jump_count = nullptr;  // K4 debug: line moved differently from neighbors
    void* d_k4_refined_sync_like_count = nullptr; // K4 debug: refined location lands in sync-like region
    void* d_k5_coarse_bad_geom_line_count = nullptr; // K5 debug: coarse-lineloc suspicious geometry count
    void* d_k5_coarse_sync_like_pixel_count = nullptr; // K5 debug: coarse-lineloc active-picture sync-like pixels
    void* d_k5_oob_pixel_count = nullptr;   // K5 debug: OOB fallback pixel count
    void* d_k5_bad_geom_line_count = nullptr; // K5 debug: suspicious lineloc geometry count
    void* d_k5_sync_like_pixel_count = nullptr; // K5 debug: in-bounds sync-like pixels
    void* d_k5_sync_like_line_counts = nullptr; // K5 debug: per-line active-picture sync-like counts

    // Kernel 7: Dropout detection output (TBC-mapped dropout entries)
    void* d_do_lines = nullptr;    // dropout line indices (batch x MAX_DROPOUTS_PER_FIELD)
    void* d_do_starts = nullptr;   // dropout start columns
    void* d_do_ends = nullptr;     // dropout end columns
    void* d_do_count = nullptr;    // dropout count per field

    // Pre-scanned field start offsets (sample positions in raw file)
    std::vector<size_t> field_offsets;

    // Host-side K2b cadence state carried across processed fields.
    int k2b_prev_first_field = -1;
    int k2b_prev_progressive_field = -1;
    long long k2b_prev_first_hsync_readloc = -1;
    double k2b_prev_first_hsync_loc = -1.0;
    double k2b_prev_first_hsync_diff = -1.0;

    size_t bytes_per_field() const;
    bool allocate_buffers();
    void free_buffers();

    // Pre-scan via FM demod: tiles the raw file, demodulates each tile,
    // and finds VSYNC positions in the clean demod domain using K3's
    // state machine. This works on any capture hardware (unlike amplitude
    // based prescan which requires head-switch artifacts).
    // Populates field_offsets[].
    bool prescan_via_demod();

    // Process one chunk of raw samples. Demodulates the chunk, finds
    // field boundaries inline via VSYNC detection, then processes fields
    // through the full pipeline. Returns number of fields processed.
    int process_chunk(size_t raw_offset, int num_fields, size_t& next_raw_offset);
};
