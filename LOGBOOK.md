# cuVHS Logbook

> **Note:** This is a development scratchpad, not documentation. Entries are
> roughly chronological but not in any coherent order — a mix of design notes,
> debugging traces, and performance measurements.

## Original Design Document

### Why From Scratch

After 10 iterations of GPU optimization via monkey-patching Python functions in
vhs-decode, the best achieved was 5.7 FPS (single GPU worker) vs 8.7 FPS (6 CPU
workers). The Python field assembly pipeline has ~50 interleaved functions with
sequential state, branching, and frequent CPU-GPU data transfers. Patching
individual functions cannot overcome this.

A from-scratch C++/CUDA pipeline processes fields entirely on GPU with no Python
in the hot path. The project was named **cuVHS** (following CUDA's naming convention:
cuFFT, cuBLAS, cuDNN, etc.).

### Pipeline Overview

```
Raw samples (host) --> [bulk upload] --> GPU VRAM
  |
  +--> Kernel 1: Batched FM Demod           (cuFFT + frequency-domain filters)
  +--> Kernel 2: Find Sync Pulses           (threshold + edge detect + compaction)
  +--> Kernel 3: Classify Pulses + Linelocs (pulse classification + line assignment)
  +--> Kernel 4: Refine Line Locations      (hsync cross-correlation)
  +--> Kernel 5: TBC Resample               (cubic interpolation to 4*fsc)
  +--> Kernel 6: Chroma Decode              (color-under extract + heterodyne + ACC)
  +--> Kernel 7: Dropout Detection          (envelope threshold + concealment)
  |
  <--[bulk download]-- TBC fields (host) --> disk
```

All inter-kernel data stays in GPU VRAM. One bulk upload at start, one bulk
download at end. No Python in the loop.

### Kernel Details

**Kernel 1: Batched FM Demodulation**
- cuFFT forward (N fields x ~468k samples at 28 MHz)
- Frequency-domain bandpass, deemphasis, chroma trap
- cuFFT inverse -> baseband video
- Hilbert unwrap for FM discrimination
- Produces: demod (video) + demod_05 (sync signal)

**Kernel 2: Find Sync Pulses**
- Threshold demod_05 at sync tip level -> boolean mask
- Edge detection: diff(mask) -> rising/falling edges
- Pair edges into pulses (start, end, length)
- Stream compaction (thrust::copy_if or CUB DeviceSelect)
- Output: per-field variable-length pulse arrays, max ~800 pulses/field

**Kernel 3: Classify Pulses + Compute Line Locations (hardest)**
- Classify each pulse by width: HSYNC (~4.7us), VSYNC (~27.1us), EQ (~2.3us)
- Find field boundary: VSYNC->HSYNC transition (sequential per field, ~800 pulses)
- Compute mean line length: parallel reduction per field
- Assign line numbers: expected[line] = ref_pos + line * mean_linelen
- For each expected position, find nearest actual HSYNC
- Output: N x linelocs (525 NTSC / 625 PAL entries per field)

**Kernel 4: Refine Line Locations (Hsync Correlation)**
- Cross-correlate ~40-sample window with ideal hsync waveform
- Sub-sample peak via parabolic interpolation
- N_fields x lines_per_field x 40 = embarrassingly parallel
- Hsync template in shared memory

**Kernel 5: TBC Resample**
- Map output position to input position via linelocs
- Cubic interpolation from demod signal
- N_fields x output_line_len x output_field_lines per batch
- Memory-bound, optimize for coalesced reads

**Kernel 6: Chroma Decode (VHS Color-Under)**
- Bandpass extract chroma-under (~629 kHz NTSC, ~626 kHz PAL)
- Heterodyne up to fsc (3.58 MHz NTSC, 4.43 MHz PAL)
- Track phase detection: try both phases, pick better burst
- ACC normalization per line: burst_abs_ref / burst_rms
- Comb filtering with adjacent field (1-field dependency)
- Note: input must be in s16-equivalent range (u8 normalized during upload)

**Kernel 7: Dropout Detection**
- Compute RF envelope from demod signal
- Threshold at configurable percentage of field average
- Hysteresis for dropout start/end
- Merge nearby dropouts, filter very short ones
- Map RF positions to TBC line/column positions
- Inline concealment: replace from adjacent line

### Memory Budget

Per field @ 28 MHz NTSC:
- Raw samples: 468k x 8 bytes (float64) = 3.6 MB
- Demod (float64): 468k x 8 = 3.6 MB
- Demod_05 (float64): 468k x 8 = 3.6 MB
- Linelocs: 525 x 8 = 4 KB
- TBC luma output: 263 x 910 x 2 = 0.5 MB
- TBC chroma output: 263 x 910 x 2 = 0.5 MB
- **Total per field: ~11.6 MB**

Batch sizing is dynamic — computed at runtime from available VRAM (uses 80% of
free VRAM by default). No hardcoded GPU assumptions.

### Implementation Phases

1. **Phase 1:** FM Demodulation (K1) — foundation for everything else
2. **Phase 2:** Sync + Line Locations (K2-K3) — field boundary detection
3. **Phase 3:** Hsync Refinement + TBC Resample (K4-K5) — produces luma TBC
4. **Phase 4:** Chroma Decode (K6) — produces chroma TBC
5. **Phase 5:** Dropout Detection (K7) — quality improvement
6. **Phase 6:** Optimization — kernel fusion, CUDA streams, PCIe overlap

### Validation Strategy

Every kernel must be validated against the Python vhs-decode reference:
- Frame-by-frame PSNR/SSIM comparison (target: >60 dB PSNR)
- Pulse-by-pulse comparison for K2-K3
- Line-by-line lineloc comparison for K3-K4
- Visual inspection of test captures across NTSC/PAL, SP/LP/EP

### Open Questions

- **PAL support:** Same kernels, different constants. Parameterized at init.
- **SVHS/Betamax/other formats:** Different chroma schemes. K6 needs format-specific
  paths. K1-K5 are format-agnostic.
- **Multi-GPU:** Partition fields across GPUs. Each GPU runs full pipeline on its subset.
- **Previous-field dependency:** Chroma comb filter and some dropout compensation need
  the previous field. Process in order within each GPU, or accept 1-field latency.

---

## Project Scaffolding (2026-03-23)

### Decision: Language and Build System

Chose standalone C++17/CUDA with CMake. The project lives at
`/media/hunter/DATA/GitHub/cuVHS/`, completely independent of the Python
vhs-decode codebase.

The name **cuVHS** follows CUDA's library naming convention (cuFFT, cuBLAS, cuDNN).

### Project Structure

```
cuVHS/
├── CMakeLists.txt                    # CMake + CUDA, multi-arch (sm_60..sm_90)
├── LOGBOOK.md
└── src/
    ├── main.cpp                      # CLI entry point, arg parsing
    ├── gpu/
    │   ├── device.h                  # GPU capability detection
    │   └── device.cu                 # Auto-select best GPU, dynamic VRAM query
    ├── format/
    │   ├── video_format.h            # NTSC/PAL timing/frequency constants
    │   └── video_format.cpp          # All params derived from system + sample rate
    ├── io/
    │   ├── raw_reader.h              # Read .u8/.s16/.u16 raw RF captures
    │   ├── raw_reader.cpp            # Normalizes u8 -> s16 range during read
    │   ├── tbc_writer.h              # Write .tbc + _chroma.tbc + .tbc.json
    │   └── tbc_writer.cpp            # ld-tools compatible JSON metadata
    └── pipeline/
        ├── pipeline.h                # Orchestrator: batch sizing, data flow
        ├── pipeline.cu               # Upload -> K1..K7 -> download -> write
        ├── fm_demod.h/cu             # K1: Batched FM demodulation (cuFFT)
        ├── sync_pulses.h/cu          # K2: Sync pulse detection
        ├── line_locs.h/cu            # K3: Pulse classification + line positions
        ├── hsync_refine.h/cu         # K4: Sub-sample hsync correlation
        ├── tbc_resample.h/cu         # K5: Cubic interpolation to 4*fsc
        ├── chroma_decode.h/cu        # K6: Color-under decode + ACC
        └── dropout_detect.h/cu       # K7: Envelope-based dropout detection
```

### Design Decisions

- **Dynamic batch sizing:** `GPUDevice::max_batch_size()` queries free VRAM at
  runtime and computes how many fields fit in 80% of available memory. No hardcoded
  GPU specs. Works on any CUDA-capable GPU.
- **Multi-arch build:** CMake builds for sm_60 through sm_90 by default, covering
  Pascal through Hopper. User can override with `-DCMAKE_CUDA_ARCHITECTURES=`.
- **u8 normalization in reader:** `RawReader::read()` converts u8 input to
  s16-equivalent range `(sample - 128) * 256` at load time. This prevents the
  chroma amplitude bug discovered in vhs-decode (see vhs-decode-faster LOGBOOK.md,
  2026-03-23 entry).
- **ld-tools compatible output:** TBC files and JSON metadata follow the same schema
  used by ld-dropout-correct, ld-chroma-decoder, and tbc-video-export.
- **All kernels stubbed:** Each kernel has a header (interface + documentation) and
  a .cu file with a stub implementation that prints a message. The pipeline runs
  end-to-end with stubs, producing zeroed output.

### Initial Build and Test

**Environment:**
- Ubuntu (kernel 6.17.0-19-generic)
- GCC 13.3.0
- CUDA 12.0 (V12.0.140) via `apt install nvidia-cuda-toolkit`
- NVIDIA GeForce RTX 3090, driver 580.126.09

**Build:**
```
mkdir build && cd build && cmake .. && make -j$(nproc)
```
One compilation fix needed: missing `#include <vector>` in `tbc_writer.h`.
After that, clean build with all 14 source files compiling successfully.

**Test run** on TAPE_1 10-second clip (280 MB, 28 MHz u8 NTSC):
```
$ cuvhs -f 28 --system NTSC --overwrite ~/captures/TAPE_1/TAPE_1_10s.u8 /tmp/cuvhs_test

GPU 0: NVIDIA GeForce RTX 3090
  Compute: 8.6  SMs: 82  Warp: 32
  VRAM: 23.6 GB total, 21.5 GB free
  Shared mem/block: 48 KB  Max threads/block: 1024
  Concurrent kernels: yes  Managed memory: yes
Format: NTSC @ 28.0 MHz capture rate
  Lines/frame: 525  Line rate: 15734.264 Hz
  Samples/line: 1780  Samples/field: 468140
  Output: 910 samples/line @ 14.318180 MHz (4*fsc)
Batch size: 512 fields (11.6 MB per field, 5961.5 MB total)
Starting decode: ~598 fields in batches of 512
Decode complete: 598 fields in 5.5 seconds (54.3 FPS)
Wrote /tmp/cuvhs_test.tbc.json (598 fields)
```

Results:
- Auto-detected RTX 3090 with 21.5 GB free VRAM
- Dynamically sized batch to 512 fields (5.96 GB)
- Processed 598 fields in 2 batches
- Output: 273 MB luma TBC, 273 MB chroma TBC, 91 KB JSON metadata
- 54 FPS with stub kernels (just I/O overhead — real perf comes with kernel implementation)
- All 7 kernel stubs executed in correct order

---

## Streaming / Real-Time Decode Support (2026-03-23)

### Motivation

The VHS preservation workflow currently requires:
1. Capture raw RF to disk (can take hours for a full tape)
2. Decode raw RF to TBC (another few hours)
3. Export TBC to video (more time)

With a GPU decoder fast enough to exceed real-time, steps 1 and 2 can be
combined: pipe the capture device's output directly into cuVHS. This saves
wall-clock time and lets the operator see decoded video during capture to
catch tracking errors, tape damage, etc. immediately rather than discovering
them hours later.

### Implementation

Reworked `RawReader` to support two modes:

**File mode** (existing): Random-access reads via `pread()`. Total size known.
Used for normal decode of captured files.

**Stream mode** (new): Sequential blocking reads from stdin or a named pipe/FIFO.
No seeking, total size unknown. Used for real-time decode during capture.

Key changes:
- `open_stream(fd)` / `open_stdin()` — new constructors for stream mode
- `read_next()` — sequential read that works in both modes (replaces offset-based `read()`)
- `read_full()` — internal helper that retries partial reads from pipes (a single
  `read()` syscall on a pipe may return less than requested even when more data is
  coming; this loops until the full amount is available or EOF)
- `open()` auto-detects FIFOs via `S_ISFIFO` and switches to stream mode
- `is_stream()` / `is_seekable()` queries for the pipeline to adapt behavior

Pipeline changes:
- Stream mode uses smaller batches (max 64 fields vs 512) for lower latency
- Progress display shows field count without percentages (total unknown)
- Main loop reads sequentially until EOF instead of computing offsets

CLI: Use `-` as input_file to read from stdin. `--format` is required for stdin
since there's no file extension to auto-detect from.

### Testing

**File mode** (unchanged behavior):
```
$ cuvhs -f 28 --system NTSC --overwrite TAPE_1_10s.u8 /tmp/test
Batch size: 512 fields — 53.5 FPS
```

**Stream mode via pipe:**
```
$ cat TAPE_1_10s.u8 | cuvhs -f 28 --system NTSC --format u8 --overwrite - /tmp/test
Input: stdin (28.0 MHz, u8, streaming)
Batch size: 64 fields [stream mode]
598 fields — 47.3 FPS (streaming)
```

Both produce identical output (598 fields, 273 MB TBC files).

The intended real-time capture usage:
```
capture_device -r 28e6 | cuvhs -f 28 --system NTSC --format u8 - output
```

---

## Kernel 1: FM Demodulation (2026-03-23)

### Overview

Implemented the full FM demodulation pipeline (K1) — the foundation kernel that
everything downstream depends on. This is the most complex single kernel in the
pipeline: RF bandpass filtering, Hilbert transform, phase unwrap (FM discrimination),
deemphasis, and dual-path post-demod filtering.

### Pipeline (all GPU, zero CPU-GPU transfers per batch)

```
Raw samples (fft_size-padded)
  |
  +--> R2C FFT (cuFFT batched D2Z)
  +--> RF bandpass × Hilbert weight (element-wise, freq domain)
  +--> Expand half → full spectrum (negative freqs = 0)
  +--> C2C inverse FFT → complex analytic signal
  +--> Phase extraction (parallel atan2)
  +--> Sequential unwrap → instantaneous frequency (Hz)
  |
  +--> R2C FFT of demod signal
  +--> Apply FVideo filter → C2R IFFT → video output (d_demod)
  +--> Apply FVideo05 filter → C2R IFFT → sync output (d_demod_05)
```

### Filter Parameters (VHS NTSC SP)

All filter coefficients are computed once on CPU at init and uploaded to GPU VRAM.
Per-batch processing is 100% GPU.

**RF bandpass (pre-demod):**
- Butterworth bandpass: 500 kHz – 6.5 MHz, order 8
- Extra lowpass: 6.0 MHz, order 25
- Extra highpass: 1.2 MHz, order 20
- Combined as magnitude product: `|BPF| × |LPF| × |HPF|`
- Multiplied by Hilbert weights: DC=1, positive freqs=2, Nyquist=1

**FM deemphasis (IEC 774-1):**
- High-shelf filter inverted to create deemphasis
- Midpoint: 273,755.82 Hz, Gain: 13.9794 dB, Q: 0.462088186
- Derived from RC time constant 1.3 µs with 4:1 resistor divider

**Post-demod video (FVideo = deemph × LPF):**
- Supergaussian LPF: 6.6 MHz, order 9
- Complex-valued (deemphasis has phase response)
- Pre-scaled by 1/N for cuFFT normalization

**Post-demod sync (FVideo05 = deemph × LPF × FIR):**
- Same as FVideo plus 65-tap FIR at 0.5 MHz (Hamming window)
- Group delay (32 samples) compensated in frequency domain via `e^{jωd}`

### FFT Size and cuFFT Plans

`samples_per_field = 468140 = 2² × 5 × 23407` — the prime factor 23407 makes this
a terrible FFT size. cuFFT's Bluestein fallback requires massive workspace, causing
`CUFFT_ALLOC_FAILED` with batch=512.

**Fix:** Pad to nearest 7-smooth number: `468750 = 2 × 3 × 5⁷` (+610 samples, 0.13%).
cuFFT handles this efficiently with mixed-radix.

Three batched cuFFT plans created at init (all double-precision):
- **D2Z** (R2C): 468750 real → 234376 complex, batch up to 512
- **Z2Z** (C2C inverse): 468750 complex → 468750 complex, batch up to 512
- **Z2D** (C2R): 234376 complex → 468750 real, batch up to 512

### Memory Layout and Buffer Reuse

`d_analytic` (max_batch × fft_size × 16 bytes) serves multiple roles:
1. **Padded input** (as `double*`): raw samples zero-padded to fft_size
2. **Full spectrum** (as `cufftDoubleComplex*`): for C2C inverse
3. **Phase angles** (as `double*`): atan2 output, aliased over complex data
4. **C2R output** (as `double*`): fft_size-strided IFFT output before trimming

This reuse avoids allocating additional scratch buffers.

**Per-field VRAM budget:**
- Pipeline buffers: ~11.6 MB (raw + demod + demod_05 + pulses + linelocs + TBC)
- K1 scratch: ~14.3 MB (fft_half + analytic + post_fft)
- Total: ~25.9 MB per field → 512 fields in ~13.3 GB

### Phase Unwrap Strategy

The FM discrimination step (complex analytic → instantaneous frequency) has a
sequential dependency: `np.unwrap` requires cumulative correction tracking.

Split into two kernels:
1. **k_compute_angles** — fully parallel atan2 on all samples across all fields
2. **k_unwrap_to_hz** — one thread per field, sequential scan over ~468k samples

The parallel atan2 dominates the cost. The sequential unwrap is ~20ms for 512 fields
(simple integer comparisons and additions, no transcendentals). Future optimization:
parallel prefix sum with CUB BlockScan.

### Butterworth Magnitude via Bilinear Transform

Instead of porting scipy's `butter()` + `freqz()`, the filter magnitudes are computed
directly from the bilinear-transform formula:

- **LPF:** `|H(ω)| = 1/√(1 + (tan(ω/2)/tan(ωc/2))^(2N))`
- **HPF:** `|H(ω)| = 1/√(1 + (tan(ωc/2)/tan(ω/2))^(2N))`
- **BPF:** `S = (tan²(ω/2) - tan(ωl/2)·tan(ωh/2)) / (BW·tan(ω/2))`, `|H| = 1/√(1+S^(2N))`

These give exact magnitude for digital Butterworth filters designed via bilinear
transform — matching scipy.signal.butter output.

### Test Results

```
$ cuvhs -f 28 --system NTSC --overwrite TAPE_1_10s.u8 /tmp/cuvhs_k1_test

GPU 0: NVIDIA GeForce RTX 3090  (21.8 GB free)
FFT size: 468750 (field: 468140, pad: +610)  freq bins: 234376  batch: 512
Batch size: 512 fields (25.9 MB per field, 13285.7 MB total)
Decode complete: 598 fields in 7.5 seconds (39.9 FPS)
```

- 39.9 FPS with real K1 computation (vs 54.3 FPS with stub — FFT overhead is ~27%)
- K2–K7 still stubs (zeroed TBC output)
- No CUDA errors, no memory issues

### Next Steps

1. Validate K1 output against Python vhs-decode reference (PSNR comparison)
2. ~~Implement K2 (sync pulse detection) — consumes d_demod_05~~ **Done**
3. Parameterize filters for PAL and tape speed (currently hardcoded NTSC SP)

---

## Kernel 2: Sync Pulse Detection (2026-03-23)

### Overview

Implemented sync pulse detection (K2) — scans the sync-demodulated signal (d_demod_05)
to find all sync pulses in each field. This is the input to K3 (pulse classification +
line location assignment).

### Algorithm

Matches Python `findpulses_numba_raw()` from `lddecode/utils.py`:

1. Signal is in Hz after FM demod (d_demod_05)
2. Threshold at -20 IRE (halfway between blanking and sync tip):
   - `pulse_threshold_hz = ire0 + hz_ire × (-20)`
   - NTSC: 3,542,857 Hz
3. Sequential scan: samples ≤ threshold are "in pulse"
4. Record (start, length) on rising edge (pulse end)
5. Skip any pulse starting at sample 0 (matches Python behavior)

One CUDA thread per field — the scan is inherently sequential within a field,
but all fields in the batch run in parallel.

### IRE-to-Hz Mapping (added to VideoFormat)

VHS FM demodulation maps IRE levels to frequencies:

| Parameter | NTSC | PAL |
|-----------|------|-----|
| Hz/IRE | 7,142.857 | 7,000 |
| 0 IRE (blanking) | 3,685,714 Hz | 4,100,000 Hz |
| Sync tip IRE | -40 | -42.857 |
| Sync tip Hz | 3,400,000 | 3,800,000 |
| Pulse threshold | 3,542,857 (-20 IRE) | 3,950,000 (-20 IRE) |

Formulas (from `vhsdecode/format_defs/vhs.py`):
- NTSC: `hz_ire = 1e6/140`, `ire0 = 4.4e6 - hz_ire×100`
- PAL: `vsync_ire = -0.3×(100/0.7)`, `hz_ire = 1e6/(100+(-vsync_ire))`, `ire0 = 4.8e6 - hz_ire×100`

### Output Layout

```
d_pulse_starts[field * MAX_PULSES + i]  = sample position of pulse start
d_pulse_lengths[field * MAX_PULSES + i] = pulse length in samples
d_pulse_count[field]                    = number of pulses found
```

MAX_PULSES = 800 (NTSC has ~263 HSYNCs + ~18 EQ/VSYNC ≈ 281; 800 is generous for
noisy signals).

### Pipeline Integration

Updated `pipeline.h` and `pipeline.cu`:
- Old: single `void* d_pulses` buffer
- New: three separate buffers — `d_pulse_starts`, `d_pulse_lengths`, `d_pulse_count`
- Updated `line_locs.h/cu` stub to accept the new interface

### Test Results

```
$ cuvhs -f 28 --system NTSC --overwrite TAPE_1_10s.u8 /tmp/k2_test

Pulse counts (first 8 fields): 268 267 267 267 267 267 267 267
Decode complete: 598 fields in 7.9 seconds (37.6 FPS)
```

- 267-269 pulses per field — exactly expected for NTSC (263 HSYNCs + ~5 EQ/VSYNC)
- 37.6 FPS (vs 39.9 FPS K1-only — K2 adds negligible overhead since it's one thread/field)
- No CUDA errors

### Next Steps

1. ~~Implement K3 (pulse classification + line location assignment)~~ **Done**
2. Validate pulse positions against Python reference
3. K3 is the hardest kernel: must classify pulses by width, find field boundaries,
   compute mean line length, and assign line positions

---

## Kernel 3: Pulse Classification + Line Locations (2026-03-23)

### Overview

Implemented pulse classification and line location assignment (K3). This kernel takes
K2's raw pulse arrays and produces per-line sample positions (linelocs) that downstream
kernels use to extract individual scanlines from the demodulated signal.

### Architecture: Three-Phase GPU Pipeline

**Phase 1: Classify pulses by width** (parallel, one thread per pulse)

Width ranges (in samples at capture rate, ±0.5 µs tolerance):
- HSYNC: 4.7 µs ± 0.5 µs → ~118-146 samples at 28 MHz
- EQ: 2.3 µs ± 0.5 µs → ~50-78 samples
- VSYNC: 13.55-28.1 µs → ~379-787 samples (generous range for noisy signals)

Matches Python `get_timings()` from `lddecode/core.py`.

**Phase 2: Vblank state machine + mean line length** (one thread per field)

Simplified version of Python `run_vblank_state_machine()`:
- State transitions: HSYNC → EQ1 → VSYNC → EQ2 → HSYNC
- Records the index of the first HSYNC after the VSYNC/EQ2 section (reference pulse)
- Finds longest continuous run of HSYNC pulses
- Averages HSYNC-to-HSYNC spacing within ±5% of nominal → mean line length
- Matches Python `computeLineLen()`

**Phase 3: Line location assignment** (one thread per field, sequential)

For each output line (0 to lines_per_frame):
1. Compute expected position: `ref_position + meanlinelen × (line - ref_line)`
2. Search forward through HSYNC-only pulses for nearest within tolerance
3. Snap to real pulse position, or keep interpolated estimate if no match
4. Tolerance: `meanlinelen / 1.5` (~0.67 line lengths)

Matches Python `valid_pulses_to_linelocs()` from `vhsdecode/sync.pyx`.

### Absolute Positioning

K2 outputs absolute (batch-global) pulse positions rather than field-relative ones.
K3's linelocs are therefore absolute positions into the contiguous d_demod buffer.

This is necessary because each field's active video extends past the field's data chunk
boundary — at 28 MHz, the VSYNC appears ~93 lines into the 257-line chunk, leaving
only ~164 lines of data after it. Active video needs 263 lines, so it reads into the
next field's data.

Since the batch data is contiguous in GPU VRAM, absolute positions let K5 (TBC resample)
read across field boundaries naturally. The only edge case is the last field in a batch
(addressed in K5 with read clamping).

### VHS Tape Speed Variation

The mean line length is computed from actual HSYNC pulse spacing, not assumed from
nominal timing. On the test tape:
- Nominal: 1780 samples/line (28 MHz / 15734.264 Hz)
- Measured: 1819 samples/line (2.2% fast)

This 2.2% speed variation is typical for VHS and would cause severe horizontal distortion
if the decoder assumed nominal timing. The adaptive measurement ensures correct output
regardless of tape speed.

### VideoFormat Additions

Added vblank structure parameters:
- `num_eq_pulses`: 6 (NTSC), 5 (PAL)
- `field_lines_first`: 263 (NTSC), 312 (PAL)
- `field_lines_second`: 262 (NTSC), 313 (PAL)

### Test Results

```
$ cuvhs -f 28 --system NTSC --overwrite TAPE_1_10s.u8 /tmp/k3_test

Field 0: 268 pulses [462..466334], linelocs spacing=1819 samples/line
Field 1: 267 pulses [469971..935805], linelocs spacing=1818 samples/line
Decode complete: 598 fields in 6.2 seconds (48.5 FPS)
```

- Consistent 1818-1819 samples/line spacing across all fields
- Linelocs properly snapped to real HSYNC pulses where available
- Lines beyond field data extrapolated from mean line length
- K3 adds negligible overhead (<1 ms for 512 fields)
- No CUDA errors

### Next Steps

1. Implement K4 (hsync refinement) — sub-sample cross-correlation
2. ~~Implement K5 (TBC resample) — cubic interpolation using K3's linelocs~~ **Done**
3. Add field order detection (first/second field) for interlaced output

---

## Kernel 5: TBC Resample (2026-03-23)

### Overview

Implemented TBC resampling (K5) — the kernel that produces actual visible output. Takes
the FM-demodulated signal (in Hz domain) and resamples it to the fixed 4×fsc timebase,
converting to uint16 TBC format compatible with ld-tools.

**First decoded frame from cuVHS is working.**

### Algorithm

One CUDA thread per output pixel (263 lines × 910 samples = 239,330 per field).

For each output pixel at (field, line, col):

1. **Map to input position**: Linear interpolation between linelocs
   ```
   line_start = linelocs[active_line_start + line]
   line_end   = linelocs[active_line_start + line + 1]
   coord = line_start + (col / output_line_len) * (line_end - line_start)
   ```

2. **Catmull-Rom cubic interpolation** at `coord` in d_demod:
   ```
   a = p2 - p0
   b = 2*p0 - 5*p1 + 4*p2 - p3
   c = 3*(p1 - p2) + p3 - p0
   value = p1 + 0.5 * x * (a + x * (b + x * c))
   ```
   Matches Python `lddecode/utils.py scale()` exactly.

3. **Hz → uint16 conversion** (ld-tools compatible):
   ```
   ire = (hz_value - ire0) / hz_ire - vsync_ire
   output = clip(ire * output_scale + output_zero, 0, 65535)
   ```

### Output Scaling (added to VideoFormat)

| Parameter | NTSC | PAL |
|-----------|------|-----|
| output_zero (sync tip) | 1024 (0x0400) | 256 (0x0100) |
| output_scale | 358.4 IRE→uint16 | 376.3 IRE→uint16 |
| White (100 IRE) | 51200 (0xC800) | 54016 (0xD300) |
| active_line_start | 10 | 8 |

### Visual Comparison

First frame decoded by cuVHS matches the Python vhs-decode reference output.
Same scene content, correct geometry, proper brightness range.

Differences vs Python reference:
- Slightly different brightness (cuVHS mean ~19500 vs Python ~23400 uint16)
  — likely due to VHS-specific vs laserdisc IRE mapping differences
- No hsync refinement yet (K4) — slightly rougher horizontal alignment
- No wow compensation — future spline-based approach like Python

### Test Results

```
$ cuvhs -f 28 --system NTSC --overwrite TAPE_1_10s.u8 /tmp/k5_test

Decode complete: 598 fields in 6.8 seconds (44.0 FPS)
Output: 273 MB luma TBC, first frame visually verified
```

- 44.0 FPS with K1+K2+K3+K5 all active (vs 48.5 K1-K3 only — K5 adds ~10% overhead)
- 239K output pixels per field × 512 fields = 122M pixels per batch
- Bounds-checked: pixels beyond batch buffer clamped to output_zero
- No CUDA errors

### Next Steps

1. Investigate brightness offset vs Python reference (IRE mapping)
2. ~~Implement K4 (hsync refinement) for better horizontal alignment~~ **Done**
3. ~~Implement K6 (chroma decode) for color output~~ **Done**

---

## Kernel 4: Hsync Refinement (2026-03-23)

### Overview

Implemented sub-sample hsync refinement (K4) — improves horizontal alignment by finding
precise zero-crossing positions in the demodulated sync signal, rather than relying on
K3's pulse-width-based estimates.

### Algorithm: Two-Pass Zero-Crossing Detection

One CUDA thread per line (lines_per_frame × num_fields threads).

**Pass 1: Initial crossing at pulse threshold**
- Search ±1 µs window around K3's lineloc estimate
- Find where d_demod_05 crosses `pulse_threshold_hz` (−20 IRE)
- Linear interpolation between adjacent samples for sub-sample accuracy

**Pass 2: Adaptive midpoint crossing**
- Measure sync level: median of samples 1.0–2.5 µs after initial crossing
- Measure porch level: median of samples 8.0–9.0 µs after initial crossing
- Midpoint = (sync + porch) / 2
- Re-find crossing at midpoint level for better accuracy
- Sanity check: refined position must be within ±0.5 µs of initial estimate

The two-pass approach handles signal amplitude variation between lines — each line
gets its own sync/porch reference levels rather than assuming a fixed threshold.

**Device functions:**
- `find_zero_crossing()` — linear interpolation between adjacent samples
- `window_median()` — bubble-sort-based median for small windows (max 64 elements)

### Test Results

```
$ cuvhs -f 28 --system NTSC --overwrite TAPE_1_10s.u8 /tmp/k4_test

Decode complete: 598 fields in 7.0 seconds (42.8 FPS)
```

- 42.8 FPS (vs 44.0 K1-K3+K5 only — K4 adds ~3% overhead)
- Visual improvement in horizontal alignment vs K3-only linelocs
- No CUDA errors

---

## Kernel 6: Chroma Decode (2026-03-24)

### Overview

Implemented VHS color-under chroma decoding (K6) — extracts the heterodyned chroma signal
from the raw RF, upconverts to colorburst frequency, and produces the chroma TBC output.

### VHS Color-Under Encoding

VHS records chroma by heterodyning the colorburst signal down to a low-frequency carrier:
- NTSC: fsc (3.579545 MHz) → 629 kHz under-carrier
- PAL: fsc (4.433619 MHz) → 626 kHz under-carrier

The decoder reverses this: multiply by (fsc + chroma_under) to shift back up, then
bandpass at fsc to extract the chroma.

### Pipeline (all GPU, per-line batched cuFFT)

```
Raw RF signal (d_raw)
  |
  +--> TBC resample via linelocs (Catmull-Rom cubic, same as K5)
  +--> Heterodyne: multiply by -cos(2π × (fsc + chroma_under) / output_rate × col)
  +--> Per-line R2C FFT (cuFFT batched D2Z)
  +--> Bandpass: zero bins outside fsc ± 500 kHz
  +--> Per-line C2R IFFT (cuFFT batched Z2D)
  +--> ACC normalization: burst_abs_ref / burst_rms per line
  +--> Convert to uint16 centered at 32768
```

### Three CUDA Kernels

**k_resample_raw_het** — TBC resample + heterodyne multiplication
- One thread per output sample (chunk_lines × fft_size)
- Maps output (field, line, col) → raw RF position via linelocs
- Catmull-Rom cubic interpolation from d_raw
- Multiplied by heterodyne carrier in same kernel (fused)
- Zero-pads beyond output_line_len to fft_size

**k_chroma_bandpass** — frequency-domain bandpass
- One thread per frequency bin
- Zeros all bins outside fsc ± 500 kHz (~bandwidth_bins)

**k_chroma_acc_output** — ACC normalization + uint16 output
- One block per line (blockDim.x = 256)
- Shared-memory reduction for burst RMS measurement
- ACC scale = burst_abs_ref / burst_rms (per-line gain normalization)
- Output: `clip(chroma × fft_scale × acc_scale + 32768, 0, 65535)`

### Dynamic FFT Sizing

FFT size = next power-of-2 ≥ output_line_len:
- NTSC: 910 → 1024
- PAL: 1135 → 2048

This avoids poor cuFFT performance on non-smooth FFT sizes while keeping
transforms small (per-line, not per-field).

### VRAM-Aware Chunking

Each line of temp data requires:
- het buffer: fft_size × 8 bytes (double)
- FFT buffer: freq_bins × 16 bytes (cufftDoubleComplex)
- NTSC: 1024×8 + 513×16 = 16,400 bytes/line
- PAL: 2048×8 + 1025×16 = 32,800 bytes/line

The decoder queries `cudaMemGetInfo()` to determine available VRAM, uses 80% for
temp buffers, and processes in field-aligned chunks. Additional cap at 4096 lines
(~15 NTSC fields) to avoid cuFFT plan creation failures with very large batch counts.

Last chunk may be smaller — cuFFT plans are recreated when batch size changes.

### VideoFormat Additions

Added burst timing parameters (system-specific, not hardcoded):
- `burst_start_us`: 5.3 µs (NTSC), 5.6 µs (PAL)
- `burst_end_us`: 7.8 µs (NTSC), 7.85 µs (PAL)
- `burst_abs_ref`: 4416 (NTSC SP), 5000 (PAL)

### Test Results

```
$ cuvhs -f 28 --system NTSC --overwrite TAPE_1_10s.u8 /tmp/k6_test

Decode complete: 598 fields in 7.1 seconds (42.2 FPS)
Output: 286 MB luma TBC, 286 MB chroma TBC
```

- 42.2 FPS with K1-K6 all active (vs 42.8 K1-K5 — K6 adds minimal overhead)
- Chroma values centered at ~32768 (correct for AC chroma signal)
- Burst region shows clear sinusoidal oscillation (~28K–38K swing around 32768)
- Active video shows chroma content varying by scene color
- Ready for ld-chroma-decoder to demodulate into U/V color components

### Known Limitation

RF filter parameters in K1 (fm_demod.cu) are currently hardcoded for NTSC SP
(bandpass frequencies, deemphasis constants). PAL and LP/EP tape speeds will need
parameterized filter coefficients — planned for Phase 3.

### Next Steps

1. ~~Implement K7 (dropout detection) — last remaining kernel~~ **Done**
2. Validate against Python vhs-decode reference (chroma PSNR)
3. Parameterize K1 filters for PAL/LP/EP (Phase 3)

---

## Kernel 7: Dropout Detection + Concealment (2026-03-24)

### Overview

Implemented RF dropout detection with inline concealment (K7) — the final kernel in the
pipeline. Detects signal dropouts from the raw RF envelope, maps them to TBC line/column
positions for ld-tools compatible metadata, and conceals affected pixels by copying from
adjacent lines.

Unlike the Python vhs-decode reference (which only records metadata and defers concealment
to the separate `ld-dropout-correct` tool), cuVHS conceals inline. This eliminates a
post-processing step and is especially valuable in streaming mode where there's no TBC
file on disk to run correction against. The concealment kernel adds negligible overhead.

### Algorithm (matches Python vhs-decode doc.py)

**Detection parameters** (same defaults as Python):
- Threshold: 18% of field mean envelope (`DOD_THRESHOLD_P = 0.18`)
- Hysteresis: 1.25× (`DOD_HYSTERESIS = 1.25`)
- Merge distance: 30 RF samples (`DOD_MERGE_DIST = 30`)
- Minimum length: 10 RF samples (`DOD_MIN_LENGTH = 10`)

### Three CUDA Kernels

**k_field_mean_sq** — parallel reduction (one block per field, 256 threads)
- Computes mean(raw²) per field using warp shuffle + shared memory reduction
- Output: one `double` per field — the mean squared RF amplitude
- This is the RF envelope reference level for thresholding

**k_detect_and_map** — sequential scan (one thread per field)
- Scans raw RF signal in blocks of 16 samples (ENV_BLOCK_SIZE)
- Computes block mean squared and compares against squared thresholds
  (avoids sqrt in inner loop)
- Hysteresis state machine: dropout starts when block_mean_sq ≤ down_thresh²,
  ends when block_mean_sq ≥ up_thresh²
- Post-scan: merges nearby dropouts (< 30 samples apart), filters short ones (< 10 samples)
- Maps RF positions to TBC line/column via binary search over linelocs
- Multi-line dropouts expand to one TBC entry per affected line
- Output: per-field arrays of (line, startx, endx) in global memory

**k_conceal** — parallel concealment (one block per field, 128 threads)
- Reads the dropout map produced by k_detect_and_map
- For each dropout entry: copies pixels from adjacent line (line above, or below for line 0)
- Applies to both luma and chroma TBC buffers
- Threads distribute across dropout entries within each field

### RF Envelope Approach

The raw RF signal contains the FM carrier. Its amplitude (RMS over a small window) is
the RF envelope. During a dropout, the tape oxide is damaged/missing and the amplitude
drops to near-zero.

Rather than computing a full Hilbert envelope (which would require another FFT pass),
K7 uses block-based RMS with 16-sample blocks (~0.57 µs at 28 MHz, roughly half a
carrier cycle). This gives sub-microsecond dropout boundary resolution without storing
a separate envelope buffer.

The threshold comparison uses squared values throughout (block_mean_sq vs threshold²)
to avoid expensive sqrt calls in the scan loop.

### RF → TBC Position Mapping

Each RF-domain dropout (start_rf, end_rf) is mapped to TBC coordinates using the
linelocs computed by K3/K4:

1. Binary search linelocs to find first and last affected TBC output line
2. Scale RF offset within line to TBC column:
   `startx = (start_rf - line_start) / (line_end - line_start) × output_line_len`
3. Multi-line dropouts produce one (line, startx, endx) entry per affected line
4. Results stored in GPU global memory, downloaded to host for JSON metadata

### Pipeline Integration

New GPU buffers: `d_do_lines`, `d_do_starts`, `d_do_ends`, `d_do_count`
(MAX_DROPOUTS_PER_FIELD = 512 entries per field, ~6 KB per field — negligible).

After K7, dropout metadata is downloaded to host alongside the TBC data and recorded
via `writer.add_dropout()` for each entry. The JSON output follows the ld-tools format:

```json
{
  "dropOuts": {
    "fieldLine": [29, 30, 31],
    "startx": [92, 0, 0],
    "endx": [910, 910, 745]
  }
}
```

### Test Results

**Clean tape (10-second NTSC clip):**
```
$ cuvhs -f 28 --system NTSC --overwrite TAPE_1_10s.u8 /tmp/k7_test

Decode complete: 598 fields in 7.4 seconds (40.3 FPS)
Fields with dropouts: 0
```
Zero dropouts — confirmed by manual inspection: minimum block RMS (8732) is well above
the 18% threshold (2587). This tape section is genuinely clean.

**Synthetic dropout injection:**
Injected a 5000-sample silence (RF amplitude = 0) at byte offset 5,000,000 in the
test clip. K7 correctly detected and mapped it:

```
Field 10: 3 dropout entries
  line 29: cols 92-910
  line 30: cols 0-910  (full line)
  line 31: cols 0-745
```

Concealment verified: all dropout pixels in lines 29-31 replaced with values from
line 28 (adjacent clean line). Exact match confirmed for columns 200-800.

Note: FM demod ringing causes minor artifacts on line 28 (adjacent to dropout) that
aren't detected as dropouts because the RF envelope stays above threshold. This matches
Python vhs-decode behavior — it's an inherent limitation of RF-domain detection.

**30-second random section (~60% into tape):**
```
$ cuvhs -f 28 --system NTSC --overwrite tape1_30s_random.u8 /tmp/k7_test2

Decode complete: 1794 fields in 20.4 seconds (44.0 FPS)
Fields with dropouts: 0
```
Also clean — this tape is in good condition.

### Performance

40.3 FPS with all 7 kernels active (vs 42.2 FPS K1-K6 only). K7 adds ~5% overhead,
dominated by the sequential per-field scan through 468K raw samples. The concealment
kernel is essentially free.

### What K7 Does NOT Detect

- **Tracking errors**: Timing misalignment where the head reads edge of adjacent track.
  RF envelope stays strong, signal is just wrong. Visible as horizontal streaks.
- **Head switching transients**: Brief disturbance at head rotation boundary.
  Same issue — signal present but discontinuous.
- **Chroma crosstalk**: Adjacent-track color leakage. Not an amplitude dropout.

These would require different detection strategies (timing analysis, not envelope
thresholding).

### tbc-video-export Flag Reference (Lesson Learned)

**Flags:**
- `-s N` / `--start N` — start at **frame** N (NOT field N)
- `-l N` / `--length N` — export N **frames**
- `--ffll N` — first visible field line (default 20 for NTSC, range 1-259)
- `--lfll N` — last visible field line (default 259 for NTSC, range 1-259)

**Correct usage for exporting frame 5:**
```bash
tbc-video-export -s 1 -l 5 output.tbc export.mkv
ffmpeg -y -i export.mkv -frames:v 5 frame%02d.png
# frame05.png = frame 5
```

**The mistake:** `-s` was assumed to be a field number. `-s 9 -l 2` was used
thinking it would select fields 9+10 (= frame 5). In reality `-s 9` starts at
**frame 9**, so the export contained frames 9-10, not frame 5. Every A/B
comparison during the K4 hsync bad-line fix work (v1 through v7) compared
frame 9 against frame 5 — completely different video content. This made it
impossible to evaluate whether code changes helped or hurt.

**Proven empirically:**
```
-s 9 -l 2, ffmpeg frame 0  =  -s 1 -l 10, ffmpeg frame 9   (bitwise identical)
-s 9 -l 2, ffmpeg frame 0  ≠  -s 1 -l 5, ffmpeg frame 5    (completely different)
```

**Rule:** Always use `-s 1 -l N` and select by ffmpeg frame number. Never use
`-s` to seek to a specific field — it's frame-based.

### Next Steps

1. Validate full pipeline against Python vhs-decode reference (PSNR comparison)
2. Parameterize K1 filters for PAL/LP/EP (Phase 3)
3. Optimization: CUDA streams, kernel fusion, PCIe overlap (Phase 6)

---

## K4 Hsync Refinement: Right-Edge Detection (2026-03-24)

### Problem

The v0 K4 implementation (left-edge only) produced visible horizontal line displacement
artifacts — individual scanlines shifted left/right by a few pixels, creating jagged
vertical edges. This was most visible on the black/white stripe pattern and on the left
edge of frame where chroma misalignment shifted into the visible area.

Multiple downstream-fix approaches were attempted and rejected:
- Median spacing smoothing (v4): degraded more lines than it fixed
- Isolated-outlier correction (v5): philosophy rejected — should fix detection, not patch symptoms
- Front porch fallback measurement: only changed 4 rows (0.7% of pixels), no visible improvement

User's guiding principle: **"I should focus on improving the detection quality rather
than attempting fixes downstream."**

### Root Cause Analysis

The v0 algorithm detected only the **leading (falling) edge** of the hsync pulse — where
the signal drops from blanking level into the sync tip. This edge is susceptible to
FM demodulation overshoot: the sharp transition from blanking to sync tip creates ringing
artifacts in the demodulated signal that shift the apparent zero-crossing position.

The Python vhs-decode reference (`refine_linelocs_hsync()` in `sync.pyx`) detects **both**
edges and prefers the **trailing (rising) edge** as the primary result. The code comment
explains why: *"less likely to be messed up by overshoot."*

### Solution: Right-Edge (Trailing Edge) Detection

Added rising-edge zero-crossing detection to K4, matching the Python reference algorithm.
The kernel now performs both left-edge and right-edge detection, preferring the right-edge
result when it passes validation.

**New device function: `find_rising_crossing()`**
- Finds first sample where signal rises above target (opposite of `find_falling_crossing`)
- Requires start sample to be below target (matches Python `calczc_do(..., edge=1)`)
- Linear interpolation for sub-sample accuracy

**Right-edge algorithm in `k_hsync_refine`:**

1. Search for rising edge near expected end of hsync pulse:
   - Start: `lineloc + hsync_width - 1µs` (1µs before expected trailing edge)
   - Count: `hsync_width × 2` samples
   - Target: `pulse_threshold_hz` (same −20 IRE threshold)

2. From the right crossing, derive the estimated leading edge:
   - `zc_fr = right_cross - hsync_width`

3. Measure sync and porch levels from the derived position:
   - Sync level: median of signal 1.0–2.5 µs after `zc_fr`
   - Porch level: median of signal `hsync_width + 1.0` to `hsync_width + 2.0` µs after `zc_fr`
   (Note: porch measurement differs from left-edge — measured from derived leading edge
   plus hsync width, matching Python `zc_fr + normal_hsync_length + 1µs`)

4. Refine at midpoint threshold (rising edge):
   - Find rising crossing at `(sync_level + porch_level) / 2`
   - Validate: refined must be within ±0.5 µs of initial right crossing

5. Compute final lineloc:
   - `right_cross - hsync_width + right_edge_offset`
   - Where `right_edge_offset = 2.25 × (sample_rate_mhz / 40.0)`
   - This is a calibration constant from Python (comment: "Magic value here, this seems
     to give approximately correct results")

6. Validation: right-edge result must be within ±2 µs of left-edge result (or original
   if left-edge failed)

**Result selection (matches Python):**
- If right-edge passes validation → use right-edge result (PRIMARY)
- Else if left-edge passes validation → use left-edge result (FALLBACK)
- Else → keep original lineloc from K3

### New kernel parameters

Two new parameters passed from host:
- `hsync_width_samples`: `fmt.hsync_width` (4.7 µs × sample_rate, already in VideoFormat)
- `right_edge_offset`: `2.25 × (sample_rate_mhz / 40.0)` (computed in host launcher)

### Results

Frame 5 comparison (10-second NTSC test clip):
- **v0 (left-edge only):** Visible horizontal line displacement on vertical edges
- **Right-edge:** Line displacement artifacts completely resolved in test frame

The right-edge result changes 95% of pixels compared to v0 — this is expected because
nearly every line gets a different (better) crossing point, which shifts the resampled
TBC output values across the entire image.

### What Changed (files)

- `src/pipeline/hsync_refine.cu`:
  - Renamed `find_zero_crossing()` → `find_falling_crossing()` (clarity)
  - Added `find_rising_crossing()` device function
  - `k_hsync_refine`: added right-edge detection with left-edge fallback
  - Host launcher: passes `hsync_width` and `right_edge_offset`

### Pending

- 5-minute clip sanity check (broader test across varied content)
- Chroma quality improvements (burst deemphasis, comb filter, track auto-detection)

---

## K6 Chroma: Track Detection + Phase Cycling Diagnosis (2026-03-24)

### Track Auto-Detection (Working)

VHS records chroma with a per-track phase rotation: NTSC uses [-1, 1] rotation
pattern, meaning even fields and odd fields have different heterodyne phase.
Getting this wrong produces no chroma or inverted chroma.

Implemented track auto-detection by trial:
1. Process field 0 with track=0 and track=1 (full resample + het + FFT + bandpass + IFFT)
2. Measure burst cancellation: sum of |burst_a + burst_b| for adjacent line pairs
3. Correct track gives lower metric (NTSC burst alternates 180°/line → cancels)
4. Typical metrics: ~700 (correct) vs ~4600 (wrong) — clear separation

Track alternates per field: `field_track[f] = (f & 1) ? (1 - detected) : detected`

### Continuous Heterodyne Phase (Fixed)

The v0 chroma kernel computed the heterodyne carrier per-line as:
```
het = -cos(2π × het_freq/output_rate × col)
```
where `col` resets to 0 each line. This creates a 171.5° phase discontinuity at
every line boundary because `output_line_len × het_freq/output_rate` is not an integer.

Python generates the carrier continuously across the full field:
```python
t = np.arange(fieldlen)
chroma_het_cos = np.cos(2π × chroma_het_freq × t / sample_rate)
```

Fixed by using absolute sample position within the field:
```
abs_sample = out_line * output_line_len + col
het = -cos(2π × het_scale × abs_sample + phase_offset)
```

### Chroma Phase Cycling (Diagnosed, Not Yet Fixed)

**Symptom:** On a 3-minute mid-tape clip, chroma exhibits a repeating cycle:
color → tearing → grayscale → tearing → color. Period is roughly every ~80-100
fields. The 10-second clip from tape start doesn't show this because it's too
short to see the cycle.

**Root Cause:** The raw RF signal fed to the heterodyne stage contains the full
spectrum — including the luma FM carrier at 3.4–4.4 MHz. When heterodyned by
(fsc + chroma_under) ≈ 4.2 MHz, the FM carrier lands right in the fsc band
(3.58 MHz), contaminating the chroma output. This contamination creates a beat
frequency that manifests as the cycling pattern.

**What Python does differently:** Before heterodyning, Python applies a Butterworth
bandpass filter on the raw RF signal at the *capture* sample rate (28 MHz):
- `get_chroma_bandpass()` in `chromaAFC.py`: 4th-order Butterworth
- Passband: 60 kHz to ~1.2 MHz (just the color-under band around 629 kHz)
- This removes the FM luma carrier entirely before it can contaminate chroma

The relevant Python call chain:
```
process.py: out_chroma = demod_chroma_filt(chroma_source, FVideoBurst, ...)
chromaAFC.py: get_chroma_bandpass() → 60 kHz to chroma_bpf_upper (1.2 MHz)
format_defs/vhs.py: chroma_bpf_upper = 1200000
```

**Fix Plan:** Add pre-bandpass filtering of raw RF around the color-under band
(60 kHz to 1.2 MHz at capture rate) before TBC resampling and heterodyning.
This requires an FFT at the capture-rate sample size (per-field, ~468k samples),
not the short per-line FFT currently used. Options:
- Reuse K1's cuFFT D2Z/Z2D plans with a new filter kernel
- Add a separate bandpass FFT pass before the existing resample+het step

### NTSC Phase Sequence (Implemented, Disabled)

Code exists for NTSC 4-frame (8-field) phase rotation via burst phase quadrant
measurement and lookup table (`ntsc_phase_table[]`). Currently disabled
(`h_phase_offset[f] = 0` for all fields) because it cannot be validated until
the pre-bandpass fix eliminates the contamination cycling.

### Progress Bar Fix

Added `fflush(stderr)` after all fprintf progress output in `pipeline.cu`.
Without this, the progress bar and dashboard were only visible after decode
completion because stderr was line-buffered when piped to a terminal emulator.

### What Changed (files)

- `src/pipeline/chroma_decode.cu`:
  - Added per-field `field_track[]` and `field_phase_offset[]` GPU arrays
  - Track auto-detection via `process_one_field_chroma()` + `measure_burst_cancellation()`
  - Burst phase measurement via `measure_burst_phase()` (I/Q product detection)
  - NTSC phase lookup table `ntsc_phase_table[]` + `lookup_ntsc_phase_offset()`
  - Continuous heterodyne phase using absolute sample position
  - Debug output to stderr (track metrics, burst phases)
- `src/pipeline/pipeline.cu`:
  - `fflush(stderr)` after progress display updates

### Next Steps

1. **Implement pre-bandpass filtering** of raw RF at capture rate (60 kHz – 1.2 MHz)
2. Re-enable NTSC phase sequence after bandpass fix validates
3. Clean up debug fprintf statements
4. Test on full 5-minute clip

---

## GPU-Parallel Chroma Track Validation (2026-03-25)

### Problem

The chroma decoder's track validation was the pipeline's primary bottleneck. The original
implementation processed each field **one at a time** in a serial loop:

1. `process_one_field_chroma()` — launches kernels for 1 field, creates/destroys cuFFT plan
2. `cudaMemcpy` — downloads ~1.5 MB chroma data to host
3. `measure_burst_cancellation()` — CPU measurement on downloaded data
4. If bad metric, repeat with flipped track

With ~411 fields per batch, that was **411 sequential GPU→CPU round-trips** per batch,
each with cuFFT plan create/destroy overhead. The actual computation was trivial — the
latency of 411 round-trips dominated.

### Fix (implemented across two sessions)

**Session 1: Eliminated the serial loop.** Restructured `chroma_decode()` Step 4 to
process all fields in batch-parallel chunks (het+FFT+BPF+IFFT), then added a
`k_burst_cancellation` GPU kernel that computes per-field burst cancellation metrics
for all fields simultaneously (one CUDA block per field, 256 threads, shared-memory
reduction). Metrics downloaded in one `cudaMemcpy` (~3 KB for 411 doubles).

**Session 2: Moved metric scan to GPU.** Added `k_find_first_bad` kernel — one thread
per field checks metric against threshold, `atomicMin` finds lowest bad index. Download
reduced to **one int (4 bytes)**. CPU only reads one int and branches.

### Flow (current)

```
1. Assign expected track pattern (alternating, from carried state)
2. Upload track assignments to GPU (one memcpy)
3. Batch het + FFT + bandpass + IFFT (all fields in parallel)
4. k_burst_cancellation — one block/field, outputs per-field metric
5. k_find_first_bad — scans metrics on GPU, outputs single int
6. Download 1 int: if -1 → proceed; if field index → flip tracks, retry (max 3)
7. k_chroma_acc_output — ACC normalization + uint16 output
```

Common case (no track flips): **one batch pass, one 4-byte download, zero per-field round-trips.**

### ChromaState carry-over

`ChromaState` struct carries `current_track`, `good_metric_threshold`, and `cycle_start`
across batch boundaries via `pipeline.h`. Eliminates fresh auto-detection per batch,
which was causing periodic greyscale dropout (~7-second intervals = batch boundary).

Track parity adjustment on carry: if batch had odd field count, `current_track` flips
for the next batch (`state->current_track = (num_fields & 1) ? (1 - current_track) : current_track`).

### Performance

| Build | 3min decode | FPS |
|-------|------------|-----|
| Before (batch parallel, serial loop still present) | ~33s | ~68 FPS |
| After (batch parallel, GPU metric scan) SM 86 | 27.4s | 82.0 FPS |
| After (batch parallel, GPU metric scan) SM 86 retest | 26.7s | 84.3 FPS |
| After (batch parallel, GPU metric scan) multi-arch | 26.3s | 85.5 FPS |
| Python vhs-decode (--no_resample) | 115s | 7.8 FPS |
| Python vhs-decode (with resample) | 159s | 6.1 FPS |

Eliminating the serial round-trip loop: ~68 → ~85 FPS (~25% improvement).
Single-arch (SM 86) vs multi-arch (60–90) is noise — confirmed with retest.
No reason to restrict arch; use multi-arch for portability.
**cuVHS is ~11x faster than Python vhs-decode** on the same hardware and input.

### Remaining serial work in chroma

- **First-batch initial detection:** 2-3 calls to `process_one_field_chroma` to test
  track=0 vs track=1 on field 0 and measure burst phase. Runs once per file, trivial.
- **Step 0 bandpass filter:** Processes one field at a time through FFT on GPU (no host
  transfers). Serial GPU work, not batched. Could be batched for further speedup but
  would require a much larger temp buffer.

### Files changed

- `src/pipeline/chroma_decode.cu`:
  - Added `k_burst_cancellation` kernel (batch-parallel burst metric, shared-mem reduction)
  - Added `k_find_first_bad` kernel (GPU metric scan, atomicMin)
  - Removed serial `process_one_field_chroma` loop
  - Restructured Step 4: batch process → GPU metric check → retry on flip → ACC output
  - `ChromaState` carry-over for batch boundaries
- `src/pipeline/chroma_decode.h`: Added `ChromaState` struct
- `src/pipeline/pipeline.h`: Added `ChromaState chroma_state` member
- `src/pipeline/pipeline.cu`: Pass `&chroma_state` to `chroma_decode()`

---

## Repo Reorganization & Clean Install (2026-03-25)

Separated cuVHS from vhs-decode-faster into fully standalone repos.

### Setup

- **cuVHS:** `/media/hunter/DATA/GitHub/cuVHS/` — pure C++/CUDA, no Python dependency
  - Build: `mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc)`
  - Multi-arch build is fine (no real perf difference vs single-arch)
- **vhs-decode:** `/media/hunter/DATA/GitHub/vhs-decode/` — clean upstream clone
  - Conda env: `vhs-decode` (not `vhs-faster`)
  - Python: `pip install /media/hunter/DATA/GitHub/vhs-decode/`
  - ld-tools: `cd vhs-decode/build && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX -DUSE_QWT=OFF && make -j$(nproc) && make install`
  - ffmpeg: `conda install -n vhs-decode ffmpeg`
  - tbc-video-export: `pip install tbc-video-export` (separate pip package)

### Tool paths (updated)

```
/home/hunter/miniforge3/envs/vhs-decode/bin/ffmpeg
/home/hunter/miniforge3/envs/vhs-decode/bin/tbc-video-export
/home/hunter/miniforge3/envs/vhs-decode/bin/vhs-decode
/home/hunter/miniforge3/envs/vhs-decode/bin/ld-dropout-correct
/home/hunter/miniforge3/envs/vhs-decode/bin/ld-chroma-decoder
```

**Note:** `tbc-video-export` needs ffmpeg and ld-tools in PATH. Run with:
```bash
PATH="/home/hunter/miniforge3/envs/vhs-decode/bin:$PATH" tbc-video-export ...
```

---

## Streaming Prescan — Fix OOM on Large Files (2026-03-25)

### Problem

`prescan_field_boundaries()` loaded the entire raw file into RAM (`new uint8_t[total_samples]`)
to do a sliding window VSYNC scan. For a 192 GB tape with 125 GB system RAM, this was an
immediate `std::bad_alloc` crash.

### Fix

Replaced the single-allocation approach with a chunked streaming scan using two sliding windows:

- **Short window** (~890 samples, half a scan line): detects the VSYNC amplitude dip
- **Long window** (~23.4M samples, ~50 fields): tracks local signal baseline

VSYNC detected when `short_mean < long_mean * 0.85`. This is a relative threshold that
adapts to local signal level, handling edit points where different recording hardware may
have different DC offsets or amplitudes. No global mean computation needed — single pass.

The long window was originally ~2 fields, but this was too short — on a 2-hour tape it only
found 30,032 of ~215,000 fields because VSYNC dips barely moved the short-term average
relative to such a small baseline. Increasing to 50 fields fixed detection completely.

**Chunk sizing:** Fixed 256 MB chunks, overlapping by `long_win` samples so both windows
stay valid across boundaries. At each new chunk, windows are re-initialized from the
overlap region. Progress reported per-chunk to stderr.

**Growing window:** The long window starts with `short_win` samples and grows to `long_win`
over the first ~50 fields. This ensures detection starts at sample `short_win` (matching the
old code) rather than waiting for the long window to fill. Verified: 30s clip produces
identical field count (1757) with both old and new prescan.

### Files changed

- `src/pipeline/pipeline.cu`: Rewrote `prescan_field_boundaries()` — chunked I/O,
  dual sliding window, fixed 256 MB chunk size.

---

## Future: CPU-Only C++ Backend

**Motivation:** A CPU-only backend would eliminate the NVIDIA GPU requirement, opening
cuVHS to AMD, Intel, laptops, servers, and Apple Silicon. The core DSP (FFTs, filters,
sliding windows) is the same math — FFTW replaces cuFFT, and CUDA kernels become regular
C++ functions. The code actually gets simpler: no device/host memory split, no transfers,
no stream synchronization.

**Why it should be fast:** Python vhs-decode is bottlenecked by single-threaded CPython
and GIL contention, not the algorithms. C++ eliminates both. A single C++ thread could
likely hit 15-30 FPS (vs Python's 2-8 FPS). With multi-core parallelism, much more.

### Three parallelization approaches

**Option A: Batched (same model as GPU)**

```
Batch 1: [field 0..15] → 16 threads, each decodes one field → write TBC → next batch
Batch 2: [field 16..31] → same
```

Direct port of the current GPU architecture. Good cache locality, chroma state carries
forward naturally between batches. Downside: threads synchronize at batch boundaries
(fastest thread waits for slowest).

**Option B: File segmentation (N independent pipelines)**

```
Thread 0: fields 0..26,000      (serial within segment)
Thread 1: fields 26,001..52,000 (serial within segment)
...etc
```

Each thread is fully independent — zero synchronization during processing. Downsides:
chroma state at segment boundaries is unknown (cold start, first few fields may be wrong),
TBC output needs coordination (N temp files or preallocated regions), uneven segments if
field density varies (tape edits, blank sections).

**Option C: Work queue (best of both)**

```
Work queue: [field 0, field 1, field 2, ...]
16 threads each grab next available field, decode it, write it
```

No batch synchronization — threads never wait for each other. Better load balancing than
fixed segments (slow fields don't block others). Each field's TBC output position is known
ahead of time, so writes don't conflict. Chroma track detection needs a small serial fixup
pass after all per-field metrics are computed (same approach as the GPU `k_find_first_bad`
kernel).

**Why Option C is probably the best choice:** Batching exists on the GPU to amortize kernel
launch overhead and maximize occupancy. On CPU there's no kernel launch cost, so the
batching constraint goes away. A simple thread pool with a work queue is both simpler and
more efficient.

### Rough performance estimates (C++ with OpenMP)

| CPU class | Cores | Est. single-thread FPS | Est. parallel FPS |
|-----------|-------|------------------------|--------------------|
| Mid-range (6C/12T) | 6 | 15-30 | 60-120 |
| Good (8C/16T) | 8 | 15-30 | 80-160 |
| Elite (16C/32T) | 16 | 15-30 | 150-300+ |
| Apple Silicon (M3 Pro+) | 10-12 | 20-40 | 150-300+ |

These are speculative but grounded: each field is ~468K samples through an FFT and filters.
FFTW on a modern x86 core handles that in single-digit milliseconds. Fields are
embarrassingly parallel — zero data dependency between them.

---

## TBC Export Profile Comparison (2026-03-25)

Benchmarked `tbc-video-export` profiles on a 30-second sample (900 frames) from a 2-hour
NTSC VHS capture decoded by cuVHS. Source: `TAPE_1_cuVHS.tbc` (430,150 fields).

All exports used default settings for each profile (no manual CRF/preset tuning).
System: RTX 3090, 32 GB RAM, Ubuntu 6.17.0.

### Results

| Profile | Codec | Pixel Format | 30s Size | Encode FPS | Encode Time | Est. 2hr Size | Est. 2hr Time |
|---------|-------|-------------|----------|------------|-------------|---------------|---------------|
| `--ffv1` (default) | FFV1 | yuv422p10le | 407 MB | 44 fps | 25s | ~97 GB | ~81 min |
| `--ffv1 --8bit` | FFV1 | yuv422p | 287 MB | 95 fps | 12s | ~69 GB | ~38 min |
| `--x264` | H.264 | yuv422p | 37 MB | 31 fps | 31s | ~9 GB | ~116 min |
| `--x265` | H.265/HEVC | yuv422p | 11 MB | 37 fps | 27s | ~2.6 GB | ~97 min |

### Notes

- **FFV1 10-bit** is the vhs-decode community standard for archival. Mathematically lossless.
- **FFV1 8-bit** is also lossless at 8-bit precision, but truncates the bottom 2 bits from
  the 16-bit TBC data. For VHS content the SNR doesn't meaningfully carry 10 bits, so the
  quality difference is negligible. 2x faster encode, 30% smaller files.
- **H.264** is lossy but visually transparent at the default quality settings. Massive size
  reduction (11x vs FFV1 8-bit). Surprisingly the slowest encode — likely due to the default
  preset/CRF tbc-video-export uses.
- **H.265** is lossy, even smaller (3.4x vs H.264). Faster to encode than H.264 here, which
  is unusual — H.265 is normally slower. Again likely down to default preset differences.

### Full 2-hour export (FFV1 10-bit default)

```
Input:    TAPE_1_cuVHS.tbc (430,150 fields, 215,075 frames)
Profile:  ffv1 (yuv422p10le)
Output:   92.41 GB, 01:59:36, 110607 kbits/s
Time:     58 min 5 sec (62 fps encode)
Dropout concealments: 0 (cuVHS dropout detection not producing metadata — see investigation below)
```

The 30s sample estimated ~97 GB; actual came in at 92.41 GB. Encode ran at 62 fps (vs 44 fps
on the 30s sample — likely better throughput once ffmpeg's pipeline is saturated).

### Recommendation

- **Archival:** FFV1 10-bit (default). Keep the full TBC data intact.
- **Working copy / space-constrained archival:** FFV1 8-bit. Best speed, reasonable size.
- **Viewing / sharing:** H.264 or H.265. Tiny files, good enough for VHS content.

## isFirstField Detection Fix (2025-03-25)

### Problem

Tape transitions (edit points) cause total breakdown of the output video — image tears,
sync loss, and phase errors for everything after the transition. On a 2-hour capture,
the transition at ~1:15:55 corrupted the entire remaining ~45 minutes.

`tbc-video-export` reported 92 instances of "Both of the determined fields have
isFirstField set", confirming the field parity metadata was wrong.

### Root Cause

Field parity was assigned naively in `pipeline.cu`:
```cpp
writer.set_first_field(i % 2 == 0);  // BUG: assumes perfect alternation
```

This breaks at tape edit points where the VCR may drop or insert a field, shifting
the even/odd cadence. Everything after the glitch gets the wrong field parity, causing
tbc-video-export to interleave fields incorrectly.

### Fix (two parts)

**Part 1: VSYNC-based parity detection in K3 (`line_locs.cu`)**

Detect field parity from the VSYNC pulse pattern, matching Python vhs-decode (`sync.pyx`).
In NTSC interlaced video, first vs second field is determined by pulse spacing at the
vblank boundary:

- **Entry spacing** (primary): last HSYNC before vblank → first EQ1 pulse.
  First field ≈ 1.0 line, second field ≈ 0.5 lines.
- **Exit spacing** (fallback): last EQ2 pulse → first HSYNC after vblank.
  Same threshold (0.75 lines). Used when entry measurement is unavailable
  (e.g., vblank starts before scan window).

Added `int* d_is_first_field` output array to `line_locs()`. Returns 1 (first),
0 (second), or -1 (unknown — no valid VSYNC sequence found).

**Part 2: Alternation enforcement in `pipeline.cu`**

VSYNC-based detection can be confidently *wrong* in garbage/transition zones.
Two consecutive fields can never have the same parity in valid interlaced video.
The pipeline enforces strict alternation:

- If detection agrees with alternation (opposite of previous field): trust it.
- If detection conflicts OR is unknown: force alternation.

Detection is only useful for establishing *initial* parity or correcting after a
genuine field drop/insert at a tape edit point. It cannot override alternation.

### Result (partial)

Initial test on the 5-minute transition clip: zero "isFirstField" errors from
tbc-video-export (was 92). But video after the tape edit still stutters badly —
parity detection returns "unknown" for ~80% of post-transition fields, and the
few it detects are all classified as second-field.

### Second Root Cause: Fixed Pulse Width Thresholds

Diagnosis via pulse width dump revealed the real problem. The two recordings
on the tape have different pulse timing:

| Pulse type | Recording 1 (samples) | Recording 2 (samples) | K3 threshold range |
|------------|----------------------|----------------------|-------------------|
| EQ         | ~65                  | 79–84                | 50–78             |
| HSYNC      | ~132                 | 145–153              | 118–146           |
| VSYNC      | ~759                 | 784–791              | 379–787           |

Recording 2's EQ pulses (79–84) fall in the gap between EQ max (78) and HSYNC
min (118), so they default to HSYNC. Result: `eq=0-5` instead of `eq=24` per
field, state machine can't find HSYNC→EQ1 transition, parity = unknown.

This is expected — different VCRs, recording speeds, or tape wear produce
different absolute pulse widths. Python vhs-decode handles this with adaptive
thresholds (`core.py:get_timings()`):

1. Generous first pass to find HSYNC-like pulses (±1.75µs window)
2. Take the **median** width → `hsync_median`
3. Compute offset from nominal: `hsync_offset = hsync_median - hsync_typical`
4. Shift **all** classification ranges (EQ, HSYNC, VSYNC) by that offset

### Fix: Adaptive Pulse Classification in K3

Implementing the same approach in `k_compute_linelocs` Phase 2 (one thread per
field, sequential scan). Before running the state machine:

1. Scan all pulse lengths, collect those within a generous HSYNC-like range
2. Compute mean (cheaper than median on GPU, close enough)
3. Derive offset from nominal HSYNC width
4. Re-classify pulses with shifted thresholds for this field

Cost: one extra linear scan over ~295 pulses per field — negligible.

### Result

**Fixed.** Tested on 5-minute clip spanning tape transition at 1:15:55 (two different
recordings spliced on the same VHS tape, with 3-4 seconds of garbage between them):

| Metric                          | Before (naive `i%2`) | After (adaptive) |
|---------------------------------|---------------------|------------------|
| Same-parity violations (total)  | 415                 | 92               |
| Pre-transition violations       | 0                   | 0                |
| Transition garbage zone         | n/a                 | 88               |
| Post-transition clean video     | hundreds            | 4                |
| tbc-video-export errors         | 92 "Both fields"    | 0                |
| tbc-video-export result         | broken/unwatchable  | clean export     |

The 88 violations in the transition zone are the actual 3-4 seconds of tape noise/garbage
visible on the original VHS — no valid video signal exists there.

## A100 Benchmark + Remove Batch Size Cap (2026-03-25)

Tested on NVIDIA A100-SXM4-80GB: **101 FPS** (~3.4x real-time NTSC) on a 5-minute clip.
About 20% faster than RTX 3090 (85 FPS), likely from the A100's higher memory bandwidth
(2 TB/s vs 936 GB/s). Prescan was slower (53s vs 12s) due to weaker single-threaded CPU
on the cluster node and network filesystem latency.

Batch size was capped at 512 by a hardcoded limit, leaving 55 GB of the A100's 80 GB VRAM
unused. Removed the cap — the VRAM budget (`gpu.max_batch_size()` at 80% of free VRAM)
is the real constraint. Stream mode retains a 64-field cap for latency.

### Performance scaling with VRAM utilization

| Run | VRAM ceiling | Batch size | VRAM used | FPS   | Real-time |
|-----|-------------|------------|-----------|-------|-----------|
| 1   | 80% (capped 512) | 512   | ~24 GB    | 101   | 3.4x      |
| 2   | 80% (uncapped)   | 1372  | ~59 GB    | 133.5 | 4.5x      |
| 3   | 95% (try-and-backoff) | 1629 | ~76 GB | 137.6 | 4.6x      |

Bigger batches help — going from 512→1372 (2.7x more fields) gave 32% more FPS. But
1372→1629 (only 19% more fields) gave just 3% more FPS, suggesting we're approaching a
throughput ceiling where memory bandwidth or compute saturation dominates over batch
overhead. The A100's 108 SMs are likely near full utilization at batch ~1400.

This is good news for consumer GPUs: an RTX 3080 (10 GB) fitting ~180 fields per batch
should still land in the efficient region of the curve. The per-field cost is mostly
fixed (FFT, demod, sync detection) — batch size primarily affects plan setup amortization
and kernel launch overhead, both of which flatten out quickly.

### Multi-GPU future

The pipeline is embarrassingly parallel at the field level — each field is independent
after prescan. Multi-GPU support would split the field offset list across devices, each
running its own batch loop. Main challenges: (1) prescan still runs on CPU, so it's
shared; (2) TBC output must be written in field order, requiring a merge step or
coordinated sequential writes; (3) chroma state (track, phase cycle) is carried across
fields, so either one GPU leads chroma decisions or each GPU handles a contiguous
segment. For a DGX with 8x A100s, even naive partitioning (split file into 8 segments)
could theoretically approach 1000 FPS — though I/O would become the bottleneck long
before that.

VRAM allocation currently uses a fixed 80% ceiling — see next entry.

## Try-and-Backoff VRAM Allocation (2026-03-25)

Replaced the hardcoded 80% VRAM fraction with a try-and-backoff loop. The pipeline now
starts at 95% of free VRAM, attempts all `cudaMalloc`s + cuFFT plan creation, and if
anything fails it frees everything and retries at 90%, 85%, etc. down to 50%.

**Why:** The old 80% was conservative — it left headroom for cuFFT workspace that we
can't predict exactly (depends on FFT size, batch count, and driver internals). But on
big GPUs like the A100, that 20% waste is 16 GB — enough for ~270 more fields in the
batch. The try-and-backoff approach discovers the real limit empirically.

**Behavior:** On most GPUs, 95% succeeds on the first try (cuFFT workspace is a small
fraction of total). If it fails, the user sees a one-line "backing off" message and the
next attempt runs immediately. Worst case is ~10 failed attempts (~1 second of startup)
before settling at 50%.

## Incremental JSON Metadata Writes (2026-03-25)

Previously, `.tbc.json` was only written at the very end of the decode in `finalize()`.
If the process was killed mid-decode (ctrl+C, OOM, crash), the TBC files existed on disk
but the JSON metadata was lost entirely — making the TBC files useless to ld-tools.

Now `write_json()` is called after every batch. It writes to a `.tmp` file then does an
atomic `rename()`, so the JSON on disk is always complete and valid — never truncated.
`finalize()` still does the final write + flushes the TBC streams.

The field metadata vector (`field_meta`) still grows in memory for the full decode. For a
6-hour tape (~3.4M fields) this is maybe 100-200 MB — not a problem in practice, but
worth noting. A future optimization could switch to append-based JSON writing to cap
memory usage, but that would require either a streaming JSON format or rewriting the
`fields` array structure.

## Future: GPU-Resident Encode Pipeline (planned)

Goal: eliminate the ld-tools CPU chain and TBC-as-bottleneck entirely. Go from raw RF
capture to encoded video in a single pass, all on GPU.

### Current workflow (decode → export)

```
raw u8 → [GPU decode] → TBC on disk → ld-dropout-correct → ld-chroma-decoder → ffmpeg → video
                         ^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                         ~100+ GB I/O   CPU-bound, multiple piped processes
```

### Planned workflow

```
raw u8 → [GPU decode] → [GPU dropout correct] → [GPU frame assembly] → [NVENC] → video
              │
              └──async──→ [CPU thread: TBC to disk] (background, optional)
```

Everything stays in VRAM between stages. NVENC is dedicated silicon on the die — doesn't
compete with CUDA cores, so it runs in parallel with the next batch's decode. The TBC
write happens via `cudaMemcpyAsync` to a host buffer + a background CPU writer thread.

### What needs to be built

1. **GPU dropout correction** — we already detect dropouts; correction (interpolate from
   adjacent lines/fields) is a straightforward kernel.
2. **GPU frame assembly** — interleave two fields into one frame using parity metadata we
   already compute. Simple copy/interleave kernel.
3. **NVENC integration** — NVIDIA Video Codec SDK, feed device pointers directly to the
   encoder. No host roundtrip.
4. **Async TBC writer** — background thread with a small ring buffer of host-pinned
   memory. Fire-and-forget, won't stall the GPU unless disk is catastrophically slow.

### Expected performance

NVENC can encode 1080i H.264/H.265 at thousands of FPS — it won't be the bottleneck.
The decode is the slowest stage (~137 FPS fields on A100, ~85 on 3090). So total
throughput should be roughly equal to current decode FPS. A 6-hour tape in ~46 minutes
on an A100, one command, no intermediate files.

### Flags

- Default: encode video + write TBC in background
- `--no-tbc`: skip TBC write for max speed / slow storage
- `--tbc-only`: current behavior, no encode (for ld-analyse users)

---

## Prescan Chunk Overlap Bug (2026-03-26)

### Symptom

47 duplicate fields in the output — cuVHS produced 11109 fields vs vhs-decode's 11060
for the same 3-minute clip. Visual comparison showed a backward time-jump at field 610:
the video rewound ~47 fields and re-decoded content it had already output.

### Root Cause

The prescan reads the u8 file in 256 MB chunks, with each chunk overlapping the previous
by `long_win` (50 fields / ~24M samples) so the sliding window has enough history. **Two
bugs in the chunked scan:**

1. **Double scan**: each chunk scanned all the way to `n_read` (which includes the overlap
   tail), but the next chunk also scanned that same overlap region as its prefix. Every
   type-A VSYNC in the overlap zone was detected twice.

2. **Unsigned underflow in dedup**: the duplicate filter `(abs_pos - last_end) > spf/2`
   used unsigned subtraction. When the next chunk re-scanned from a lower absolute
   position than the previous chunk ended at, the subtraction wrapped to ~2^64 and the
   filter passed. The guard was always true for duplicates instead of always false.

### Fix

1. Limit each chunk's scan loop to only its new samples (`scan_end_in_buf`), not the
   overlap tail that exists purely for window context.
2. Add `abs_pos > last_end` guard before the unsigned subtraction.

### Result

- 11062 prescan offsets → 11061 decoded fields (vs vhs-decode's 11060)
- The 1-field difference is the field-0-at-offset-0 convention
- Zero backward jumps, zero duplicate offsets
- The field 608-612 time-travel glitch is eliminated

---

## K3 Wrong-VSYNC Fix: Tape Edit Tearing + Chroma Corruption (2026-03-26)

### Symptom

~200 fields near a tape edit (fields 5572-5776 in a 3-minute test clip) showed horizontal
wrapping and vertical displacement in the TBC output. The image content was correct but
positioned wrong — the entire field was shifted ~250 output samples horizontally. vhs-decode
decoded the same region cleanly.

A secondary symptom: when the fix was first applied without a guard, it corrupted field 0,
which poisoned the chroma track detector's carried state and caused chroma to cycle in and
out (greyscale dropout) across the entire file.

### Root Cause

K3's VSYNC state machine scans forward through the pulse list looking for the pattern
EQ1→VSYNC→EQ2→HSYNC. Each field chunk contains two VSYNCs: this field's (near the start)
and the next field's (near the end). Normally the state machine finds the first one at
pulse ~20 and stops.

Near tape edits, the leading sync pulses are corrupted — the EQ/VSYNC/EQ pattern doesn't
match. The state machine walks right past the damaged first VSYNC, keeps scanning, and locks
onto the **next field's** clean VSYNC near pulse ~280. It then lays out all 525 lines
anchored from a reference point that's ~477K samples (one full field) too far into the chunk.

Instrumented K3 output confirmed: good fields have `ref_pulse_idx ≈ 20`, bad fields have
`ref_pulse_idx ≈ 280-290`. All other K3 outputs (meanlinelen, best_run_len, state) were
identical between good and bad fields.

### Fix

In `k_compute_linelocs()`, after the state machine selects a reference pulse:

```cpp
if (ref_pulse_idx > npc_clamped / 2) {
    double corrected = ref_position - (double)samples_per_field;
    if (corrected > 0.0) {
        ref_position = corrected;
    }
}
```

If the reference pulse is past the midpoint of the pulse list, it found the wrong VSYNC.
Subtract `samples_per_field` to project back to the correct position. The `corrected > 0`
guard prevents false correction on field 0, which legitimately has its VSYNC in the middle
of the chunk (it starts before the first VSYNC in the file).

### Why the guard matters

Without the `corrected > 0` check, field 0 (which has `ref_pulse_idx ≈ 155` because the
field starts before any VSYNC) gets a negative ref_position. This corrupts field 0's line
positions, which causes the chroma track detector to misidentify the initial track. Since
chroma state is carried across batches, the bad initial state propagates through the entire
file, causing periodic chroma dropout.

### Result

- All ~200 previously-torn fields now decode with correct horizontal/vertical alignment
- Chroma remains stable throughout (no greyscale cycling)
- No measurable performance impact (the fix is 5 lines in a per-field kernel)
- vhs-decode handles this differently: it validates each field against the previous one's
  timing, which provides implicit temporal coherence. The K3 fix achieves the same result
  without cross-field state, preserving full parallelism.