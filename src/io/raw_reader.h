#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include "format/video_format.h"

// Reads raw RF captures from disk. Supports .u8, .s16, .u16.
// Data is read in large chunks and converted to float64 for the pipeline.
struct RawReader {
    ~RawReader();

    bool open(const std::string& path, InputFormat fmt);
    void close();

    // Read `num_samples` starting at `offset` (in samples, not bytes).
    // Converts to double and normalizes:
    //   u8:  (sample - 128) * 256  → s16-equivalent range
    //   s16: passthrough as double
    //   u16: (sample - 32768) as double
    // Returns number of samples actually read (may be less at EOF).
    size_t read(double* dest, size_t offset, size_t num_samples);

    // Read directly into a host buffer without conversion (for bulk GPU upload).
    // Caller handles conversion on GPU.
    size_t read_raw(void* dest, size_t offset, size_t num_samples);

    size_t total_samples() const;
    size_t size_bytes() const { return file_size; }
    InputFormat format() const { return fmt; }

private:
    int fd = -1;
    InputFormat fmt = InputFormat::U8;
    size_t file_size = 0;
};
