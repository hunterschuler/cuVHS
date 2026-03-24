#pragma once
#include <cstddef>
#include <cstdint>

struct GPUDevice {
    int device_id = -1;
    char name[256] = {};
    size_t vram_total = 0;        // bytes
    size_t vram_free = 0;         // bytes at init time
    int sm_count = 0;             // streaming multiprocessors
    int compute_major = 0;
    int compute_minor = 0;
    int max_threads_per_block = 0;
    int warp_size = 0;
    size_t shared_mem_per_block = 0;
    bool supports_managed_memory = false;
    bool supports_concurrent_kernels = false;

    // Initialize GPU. id=-1 auto-selects the best available GPU.
    bool init(int id = 0);

    // Print GPU info to stderr.
    void print_info() const;

    // Query current free VRAM (call anytime, not just at init).
    size_t query_free_vram() const;

    // Compute how many fields can fit in a given fraction of free VRAM.
    // bytes_per_field = total GPU memory needed per field (all buffers).
    // vram_fraction = how much of free VRAM to use (default 0.8 = 80%).
    int max_batch_size(size_t bytes_per_field, double vram_fraction = 0.8) const;
};
