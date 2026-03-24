#pragma once
#include "gpu/device.h"
#include "format/video_format.h"
#include "io/raw_reader.h"
#include "io/tbc_writer.h"

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

    // GPU buffer pointers (allocated in run(), freed in destructor)
    // These are device pointers.
    void* d_raw = nullptr;         // raw input samples (batch)
    void* d_demod = nullptr;       // FM demod output (float64, batch)
    void* d_demod_05 = nullptr;    // sync demod output (float64, batch)
    void* d_pulses = nullptr;      // pulse arrays (batch)
    void* d_linelocs = nullptr;    // line locations (batch)
    void* d_tbc_luma = nullptr;    // TBC luma output (uint16, batch)
    void* d_tbc_chroma = nullptr;  // TBC chroma output (uint16, batch)

    size_t bytes_per_field() const;
    bool allocate_buffers();
    void free_buffers();

    // Process one batch of fields. Reads sequentially from the reader.
    // Returns number of fields processed (0 at EOF).
    int process_batch(int num_fields);
};
