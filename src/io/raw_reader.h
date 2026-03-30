#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include "format/video_format.h"

// Reads raw RF captures. Two modes:
//
//   File mode:  Open a file by path. Supports random access (seek + read).
//               Total size is known upfront. Used for normal decoding.
//
//   Stream mode: Read from stdin or a pipe/FIFO. Sequential reads only,
//                no seeking, total size unknown. Used for real-time decode
//                during live capture. Blocks until data is available or EOF.
//
// Both modes convert samples to float64 and normalize to s16-equivalent range:
//   u8:  (sample - 128) * 256
//   s16: passthrough
//   u16: (sample - 32768)
//
struct InputConditioning {
    bool dc_correct = false;  // subtract per-read mean after bit-depth normalization
};

struct RawReader {
    ~RawReader();

    void set_conditioning(const InputConditioning& cfg) { conditioning = cfg; }
    InputConditioning get_conditioning() const { return conditioning; }

    // File mode: open a regular file by path.
    bool open(const std::string& path, InputFormat fmt);

    // Stream mode: read from an existing fd (e.g., STDIN_FILENO).
    // The fd is NOT closed by this class.
    bool open_stream(int fd, InputFormat fmt);

    // Stream mode convenience: open stdin.
    bool open_stdin(InputFormat fmt);

    void close();

    // --- File mode API (random access) ---

    // Read `num_samples` starting at `offset` (in samples).
    // Returns number of samples read. Only works in file mode.
    size_t read_at(double* dest, size_t offset, size_t num_samples);

    // Read raw bytes at offset without conversion. File mode only.
    size_t read_raw_at(void* dest, size_t offset, size_t num_samples);

    // --- Stream mode API (sequential) ---

    // Read the next `num_samples` from the stream. Blocks until all
    // samples are available, or returns fewer at EOF/error.
    // Works in both modes (file mode reads sequentially from current pos).
    size_t read_next(double* dest, size_t num_samples);

    // Read raw bytes sequentially without conversion.
    size_t read_next_raw(void* dest, size_t num_samples);

    // --- Info ---

    // Total samples in file. Returns 0 in stream mode (unknown).
    size_t total_samples() const;

    // Total bytes of input. Returns 0 in stream mode.
    size_t size_bytes() const { return file_size; }

    InputFormat format() const { return fmt; }
    bool is_stream() const { return streaming; }
    bool is_seekable() const { return !streaming; }

private:
    int fd = -1;
    bool owns_fd = false;        // true if we opened it, false if caller owns it
    bool streaming = false;
    InputFormat fmt = InputFormat::U8;
    InputConditioning conditioning;
    size_t file_size = 0;        // 0 for streams

    // Convert raw samples in `buf` to normalized float64 in `dest`.
    void convert(const void* buf, double* dest, size_t num_samples) const;

    // Read exactly `byte_count` bytes from fd, retrying on partial reads.
    // Returns bytes actually read. For pipes, this is essential since
    // a single read() may return less than requested even when more data
    // is coming.
    static ssize_t read_full(int fd, void* buf, size_t byte_count);
};
