#include "io/raw_reader.h"
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <algorithm>

RawReader::~RawReader() {
    close();
}

bool RawReader::open(const std::string& path, InputFormat format) {
    fmt = format;
    streaming = false;

    fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        perror(path.c_str());
        return false;
    }
    owns_fd = true;

    struct stat st;
    if (fstat(fd, &st) < 0) {
        perror("fstat");
        close();
        return false;
    }

    // Detect if this is a pipe/FIFO vs regular file
    if (S_ISFIFO(st.st_mode) || S_ISCHR(st.st_mode)) {
        streaming = true;
        file_size = 0;
    } else {
        file_size = st.st_size;
    }

    return true;
}

bool RawReader::open_stream(int stream_fd, InputFormat format) {
    fmt = format;
    fd = stream_fd;
    owns_fd = false;
    streaming = true;
    file_size = 0;
    return fd >= 0;
}

bool RawReader::open_stdin(InputFormat format) {
    return open_stream(STDIN_FILENO, format);
}

void RawReader::close() {
    if (fd >= 0 && owns_fd) {
        ::close(fd);
    }
    fd = -1;
    owns_fd = false;
}

size_t RawReader::total_samples() const {
    if (streaming) return 0;
    return file_size / input_format_bytes_per_sample(fmt);
}

// --- Conversion ---

void RawReader::convert(const void* buf, double* dest, size_t num_samples) const {
    switch (fmt) {
        case InputFormat::U8: {
            auto* src = static_cast<const uint8_t*>(buf);
            for (size_t i = 0; i < num_samples; i++)
                dest[i] = static_cast<double>(src[i]);
            break;
        }
        case InputFormat::S16: {
            auto* src = static_cast<const int16_t*>(buf);
            for (size_t i = 0; i < num_samples; i++)
                dest[i] = static_cast<double>(src[i]);
            break;
        }
        case InputFormat::U16: {
            auto* src = static_cast<const uint16_t*>(buf);
            for (size_t i = 0; i < num_samples; i++)
                dest[i] = static_cast<double>(src[i]) - 32768.0;
            break;
        }
    }
}

// --- Retry helper for pipes ---

ssize_t RawReader::read_full(int fd, void* buf, size_t byte_count) {
    size_t total = 0;
    auto* p = static_cast<uint8_t*>(buf);

    while (total < byte_count) {
        ssize_t n = ::read(fd, p + total, byte_count - total);
        if (n > 0) {
            total += n;
        } else if (n == 0) {
            // EOF
            break;
        } else {
            // Error — retry on EINTR (signal interrupted), fail otherwise
            if (errno == EINTR) continue;
            if (total > 0) break;  // return what we have
            return -1;
        }
    }
    return static_cast<ssize_t>(total);
}

// --- File mode (random access) ---

size_t RawReader::read_at(double* dest, size_t offset, size_t num_samples) {
    if (streaming) {
        fprintf(stderr, "read_at() not supported in stream mode\n");
        return 0;
    }

    int bps = input_format_bytes_per_sample(fmt);
    size_t byte_offset = offset * bps;
    if (byte_offset >= file_size) return 0;

    size_t available = (file_size - byte_offset) / bps;
    size_t to_read = std::min(num_samples, available);
    size_t byte_count = to_read * bps;

    auto* buf = new uint8_t[byte_count];
    ssize_t n = pread(fd, buf, byte_count, byte_offset);
    if (n <= 0) {
        delete[] buf;
        return 0;
    }
    size_t samples_read = n / bps;
    convert(buf, dest, samples_read);
    delete[] buf;
    return samples_read;
}

size_t RawReader::read_raw_at(void* dest, size_t offset, size_t num_samples) {
    if (streaming) {
        fprintf(stderr, "read_raw_at() not supported in stream mode\n");
        return 0;
    }

    int bps = input_format_bytes_per_sample(fmt);
    size_t byte_offset = offset * bps;
    if (byte_offset >= file_size) return 0;

    size_t available = (file_size - byte_offset) / bps;
    size_t to_read = std::min(num_samples, available);
    size_t byte_count = to_read * bps;

    ssize_t n = pread(fd, dest, byte_count, byte_offset);
    if (n <= 0) return 0;
    return n / bps;
}

// --- Stream mode (sequential) ---

size_t RawReader::read_next(double* dest, size_t num_samples) {
    int bps = input_format_bytes_per_sample(fmt);
    size_t byte_count = num_samples * bps;

    auto* buf = new uint8_t[byte_count];
    ssize_t n = read_full(fd, buf, byte_count);
    if (n <= 0) {
        delete[] buf;
        return 0;
    }
    size_t samples_read = n / bps;
    convert(buf, dest, samples_read);
    delete[] buf;
    return samples_read;
}

size_t RawReader::read_next_raw(void* dest, size_t num_samples) {
    int bps = input_format_bytes_per_sample(fmt);
    size_t byte_count = num_samples * bps;

    ssize_t n = read_full(fd, dest, byte_count);
    if (n <= 0) return 0;
    return n / bps;
}
