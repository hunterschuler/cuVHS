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
    void* d_linelocs = nullptr;    // line locations (batch)
    void* d_envelope = nullptr;    // RF envelope magnitude (float64, batch)
    void* d_tbc_luma = nullptr;    // TBC luma output (uint16, batch)
    void* d_tbc_chroma = nullptr;  // TBC chroma output (uint16, batch)
    void* d_is_first_field = nullptr;  // field parity from VSYNC pulse pattern (int, batch)

    // Kernel 7: Dropout detection output (TBC-mapped dropout entries)
    void* d_do_lines = nullptr;    // dropout line indices (batch x MAX_DROPOUTS_PER_FIELD)
    void* d_do_starts = nullptr;   // dropout start columns
    void* d_do_ends = nullptr;     // dropout end columns
    void* d_do_count = nullptr;    // dropout count per field

    // Pre-scanned field start offsets (sample positions in raw file)
    std::vector<size_t> field_offsets;

    size_t bytes_per_field() const;
    bool allocate_buffers();
    void free_buffers();

    // Pre-scan the raw file to find VSYNC-based field boundaries.
    // Populates field_offsets[].
    bool prescan_field_boundaries();

    // Process one batch of fields using pre-scanned offsets.
    // Returns number of fields processed (0 at EOF).
    int process_batch(int start_field, int num_fields);
};
