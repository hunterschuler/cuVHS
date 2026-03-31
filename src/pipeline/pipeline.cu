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

static bool env_flag_enabled(const char* name) {
    const char* env = getenv(name);
    return env && env[0] && strcmp(env, "0") != 0;
}

static const char* pulse_type_name(int type) {
    switch (type) {
        case PULSE_HSYNC: return "H";
        case PULSE_EQ1: return "EQ1";
        case PULSE_VSYNC: return "V";
        case PULSE_EQ2: return "EQ2";
        default: return "?";
    }
}

struct HostK3MatchDetail {
    bool valid = false;
    bool matched = false;
    int line = -1;
    int cursor_before = -1;
    int cursor_after = -1;
    int chosen_hsync_slot = -1;
    int chosen_pos = -1;
    int prev_hsync = -1;
    int next_hsync = -1;
    double expected = 0.0;
    double match_dist = 0.0;
    double max_allowed_distance = 0.0;
};

struct HostLineMarker {
    int start = 0;
    int length = 0;
    int source_idx = -1;
};

enum HostMarkerMode {
    HOST_MARKERS_BASE = 0,
    HOST_MARKERS_BRIDGED = 1,
    HOST_MARKERS_SUPPRESS_DENSE = 2,
    HOST_MARKERS_K2_DEBOUNCE = 3,
    HOST_MARKERS_K2_KEEP_FIRST = 4,
    HOST_MARKERS_K2_MERGE_CLUSTER = 5,
    HOST_MARKERS_K2_VALIDATED = 6,
};

struct HostRefTrackResult {
    bool valid = false;
    std::vector<double> locs;
};

struct HostRefImprovement {
    int field = -1;
    int line = -1;
    double current_error = 0.0;
    double ref_error = 0.0;
    double improvement = 0.0;
};

struct HostRefPatch {
    int field = -1;
    int line = -1;
    double coarse_loc = 0.0;
    double ref_loc = 0.0;
    double delta = 0.0;
    double improvement = 0.0;
};

struct HostRefDisagreement {
    int field = -1;
    int line = -1;
    double coarse_loc = 0.0;
    double ref_loc = 0.0;
    double delta = 0.0;
    double coarse_error = 0.0;
    double ref_error = 0.0;
    double improvement = 0.0;
};

static int classify_host_pulse_type(double len, const K3Debug& dbg, const VideoFormat& fmt) {
    double tol = fmt.sample_rate * 0.5e-6;
    double hsync_nominal = fmt.hsync_width + dbg.hsync_offset;
    double eq_nominal = fmt.eq_pulse_width + dbg.hsync_offset;
    double vsync_nominal = fmt.vsync_width + dbg.hsync_offset;
    double a_hsync_min = hsync_nominal - tol;
    double a_hsync_max = hsync_nominal + tol;
    double a_eq_min = eq_nominal - tol;
    double a_eq_max = eq_nominal + tol;
    double a_vsync_min = vsync_nominal * 0.5;
    double a_vsync_max = vsync_nominal + 2.0 * tol;
    if (len >= a_vsync_min && len <= a_vsync_max) return PULSE_VSYNC;
    if (len >= a_hsync_min && len <= a_hsync_max) return PULSE_HSYNC;
    if (len >= a_eq_min && len <= a_eq_max) return PULSE_EQ1;
    return PULSE_HSYNC;
}

static bool host_pulse_qualitycheck(int prev_type, int pulse_type, int start_delta, int in_line_len) {
    double min_lines = 0.0;
    double max_lines = 0.0;
    if (prev_type > 0 && pulse_type > 0) {
        min_lines = 0.4;
        max_lines = 0.6;
    } else if (prev_type == 0 && pulse_type == 0) {
        min_lines = 0.9;
        max_lines = 1.1;
    } else {
        min_lines = 0.4;
        max_lines = 1.1;
    }
    double line_span = (double)start_delta / (double)in_line_len;
    return line_span >= min_lines && line_span <= max_lines;
}

struct HostPulseRefInfo {
    bool valid = false;
    double meanlinelen = 0.0;
    double ref_position = 0.0;
    double ref_line = 10.0;
    int ref_pulse_idx = -1;
    int field_parity = -1;
};

static int classify_nominal_pulse_type(double len, const VideoFormat& fmt) {
    double tol = fmt.sample_rate * 0.5e-6;
    double hsync_min = fmt.hsync_width - tol;
    double hsync_max = fmt.hsync_width + tol;
    double eq_min = fmt.eq_pulse_width - tol;
    double eq_max = fmt.eq_pulse_width + tol;
    double vsync_min = fmt.vsync_width * 0.5;
    double vsync_max = fmt.vsync_width + 2.0 * tol;
    if (len >= vsync_min && len <= vsync_max) return PULSE_VSYNC;
    if (len >= hsync_min && len <= hsync_max) return PULSE_HSYNC;
    if (len >= eq_min && len <= eq_max) return PULSE_EQ1;
    return PULSE_HSYNC;
}

static HostPulseRefInfo estimate_host_pulse_reference(const int* pulse_starts,
                                                      const int* pulse_lengths,
                                                      int pulse_count,
                                                      const VideoFormat& fmt) {
    HostPulseRefInfo info;
    int npc = std::max(0, std::min(pulse_count, MAX_PULSES));
    if (npc < 8) return info;

    std::vector<int> types((size_t)npc);
    for (int i = 0; i < npc; i++) {
        types[(size_t)i] = classify_nominal_pulse_type((double)pulse_lengths[i], fmt);
    }

    int best_run_start = -1, best_run_len = 0;
    int cur_run_start = -1, cur_run_len = 0;
    for (int i = 0; i < npc; i++) {
        if (types[(size_t)i] == PULSE_HSYNC) {
            if (cur_run_start < 0) {
                cur_run_start = i;
                cur_run_len = 1;
            } else {
                cur_run_len++;
            }
        } else {
            if (cur_run_len > best_run_len) {
                best_run_start = cur_run_start;
                best_run_len = cur_run_len;
            }
            cur_run_start = -1;
            cur_run_len = 0;
        }
    }
    if (cur_run_len > best_run_len) {
        best_run_start = cur_run_start;
        best_run_len = cur_run_len;
    }

    double meanlinelen = (double)fmt.samples_per_line;
    if (best_run_len >= 2) {
        double sum = 0.0;
        int count = 0;
        for (int i = best_run_start + 1; i < best_run_start + best_run_len; i++) {
            double spacing = (double)(pulse_starts[i] - pulse_starts[i - 1]);
            if (spacing >= meanlinelen * 0.95 && spacing <= meanlinelen * 1.05) {
                sum += spacing;
                count++;
            }
        }
        if (count > 0) meanlinelen = sum / count;
    }
    if (!(meanlinelen > 0.0)) return info;

    int ref_pulse_idx = -1;
    int last_hsync_idx = -1;
    int first_eq1_idx = -1;
    int last_eq2_idx = -1;
    int state = -1;
    for (int i = 0; i < npc; i++) {
        int t = types[(size_t)i];
        if (state == -1) {
            if (t == PULSE_HSYNC) {
                state = PULSE_HSYNC;
                last_hsync_idx = i;
            }
        } else if (state == PULSE_HSYNC) {
            if (t == PULSE_HSYNC) {
                last_hsync_idx = i;
            } else if (t == PULSE_EQ1) {
                state = PULSE_EQ1;
                first_eq1_idx = i;
            } else if (t == PULSE_VSYNC) {
                state = PULSE_VSYNC;
            }
        } else if (state == PULSE_EQ1) {
            if (t == PULSE_VSYNC) {
                state = PULSE_VSYNC;
            } else if (t == PULSE_HSYNC) {
                state = PULSE_HSYNC;
                last_hsync_idx = i;
                first_eq1_idx = -1;
            }
        } else if (state == PULSE_VSYNC) {
            if (t == PULSE_EQ1) {
                state = PULSE_EQ2;
                last_eq2_idx = i;
            } else if (t == PULSE_HSYNC) {
                ref_pulse_idx = i;
                break;
            }
        } else if (state == PULSE_EQ2) {
            if (t == PULSE_EQ1) {
                last_eq2_idx = i;
            } else if (t == PULSE_HSYNC) {
                if (last_eq2_idx < 0) last_eq2_idx = i - 1;
                ref_pulse_idx = i;
                break;
            }
        }
    }

    int field_parity = -1;
    if (ref_pulse_idx >= 0 && last_hsync_idx >= 0 && first_eq1_idx >= 0) {
        double entry_spacing = (double)(pulse_starts[first_eq1_idx] - pulse_starts[last_hsync_idx]);
        double entry_lines = entry_spacing / meanlinelen;
        if (fmt.system == VideoSystem::NTSC) field_parity = (entry_lines > 0.75) ? 1 : 0;
        else field_parity = (entry_lines > 0.75) ? 0 : 1;
    }
    if (field_parity < 0 && ref_pulse_idx >= 0 && last_eq2_idx >= 0) {
        double exit_spacing = (double)(pulse_starts[ref_pulse_idx] - pulse_starts[last_eq2_idx]);
        double exit_lines = exit_spacing / meanlinelen;
        if (fmt.system == VideoSystem::NTSC) field_parity = (exit_lines > 0.75) ? 1 : 0;
        else field_parity = (exit_lines > 0.75) ? 0 : 1;
    }

    double ref_position = (ref_pulse_idx >= 0) ? (double)pulse_starts[ref_pulse_idx] : (double)pulse_starts[0];
    double ref_line = 10.0;

    info.valid = true;
    info.meanlinelen = meanlinelen;
    info.ref_position = ref_position;
    info.ref_line = ref_line;
    info.ref_pulse_idx = ref_pulse_idx;
    info.field_parity = field_parity;
    return info;
}

static int host_k2b_validate_pulses(const int* in_starts,
                                    const int* in_lengths,
                                    int in_count,
                                    int* out_starts,
                                    int* out_lengths,
                                    const VideoFormat& fmt) {
    int npc = std::max(0, std::min(in_count, MAX_PULSES));
    if (npc <= 0) return 0;

    HostPulseRefInfo ref = estimate_host_pulse_reference(in_starts, in_lengths, npc, fmt);
    if (!ref.valid) {
        for (int i = 0; i < npc; i++) {
            out_starts[i] = in_starts[i];
            out_lengths[i] = in_lengths[i];
        }
        return npc;
    }

    struct Candidate {
        int start = 0;
        int length = 0;
        int type = PULSE_HSYNC;
        bool keep = false;
        bool keep_vblank = false;
        int rounded_line = -1;
        double line_dist = 1e9;
        double score = 1e9;
    };

    std::vector<Candidate> candidates((size_t)npc);
    std::vector<int> best_h_idx((size_t)fmt.lines_per_frame, -1);
    double line0loc = ref.ref_position - ref.meanlinelen * ref.ref_line;
    int ref_window_lo = std::max(0, ref.ref_pulse_idx - 16);
    int ref_window_hi = std::min(npc - 1, ref.ref_pulse_idx + 16);

    for (int i = 0; i < npc; i++) {
        Candidate cand;
        cand.start = in_starts[i];
        cand.length = in_lengths[i];
        cand.type = classify_nominal_pulse_type((double)in_lengths[i], fmt);

        bool valid_prev = false;
        bool valid_next = false;
        if (i > 0) {
            int prev_type = classify_nominal_pulse_type((double)in_lengths[i - 1], fmt);
            valid_prev = host_pulse_qualitycheck(prev_type, cand.type, in_starts[i] - in_starts[i - 1], fmt.samples_per_line);
        }
        if (i + 1 < npc) {
            int next_type = classify_nominal_pulse_type((double)in_lengths[i + 1], fmt);
            valid_next = host_pulse_qualitycheck(cand.type, next_type, in_starts[i + 1] - in_starts[i], fmt.samples_per_line);
        }

        if (cand.type == PULSE_HSYNC) {
            double lineloc = ((double)cand.start - line0loc) / ref.meanlinelen;
            cand.rounded_line = (int)std::llround(lineloc);
            cand.line_dist = fabs(lineloc - (double)cand.rounded_line);
            bool in_range = cand.rounded_line >= 0 && cand.rounded_line < fmt.lines_per_frame;
            bool plausible_line = in_range && cand.line_dist <= 0.30;
            bool cadence_ok = valid_prev || valid_next;
            double width_error = fabs((double)cand.length - fmt.hsync_width);
            cand.score = cand.line_dist * 1000.0 + width_error;
            cand.keep = plausible_line && cadence_ok;
            if (cand.keep && in_range) {
                int& best = best_h_idx[(size_t)cand.rounded_line];
                if (best < 0 || cand.score < candidates[(size_t)best].score) best = i;
            }
        } else {
            bool near_vblank = (ref.ref_pulse_idx >= 0 && i >= ref_window_lo && i <= ref_window_hi);
            cand.keep_vblank = near_vblank && (valid_prev || valid_next);
        }

        candidates[(size_t)i] = cand;
    }

    int out_count = 0;
    for (int i = 0; i < npc; i++) {
        const Candidate& cand = candidates[(size_t)i];
        bool keep = false;
        if (cand.type == PULSE_HSYNC) {
            if (cand.rounded_line >= 0 && cand.rounded_line < fmt.lines_per_frame) {
                keep = best_h_idx[(size_t)cand.rounded_line] == i;
            }
        } else {
            keep = cand.keep_vblank;
        }
        if (!keep) continue;
        if (out_count >= MAX_PULSES) break;
        out_starts[out_count] = cand.start;
        out_lengths[out_count] = cand.length;
        out_count++;
    }

    if (out_count < 32) {
        for (int i = 0; i < npc && out_count < std::min(npc, MAX_PULSES); i++) {
            bool already_kept = false;
            for (int j = 0; j < out_count; j++) {
                if (out_starts[j] == in_starts[i] && out_lengths[j] == in_lengths[i]) {
                    already_kept = true;
                    break;
                }
            }
            if (already_kept) continue;
            out_starts[out_count] = in_starts[i];
            out_lengths[out_count] = in_lengths[i];
            out_count++;
        }
    }

    return out_count;
}

struct HostPulse {
    int start = 0;
    int len = 0;
};

struct HostValidPulse {
    int type = PULSE_HSYNC;
    HostPulse pulse;
    bool good = false;
};

struct HostK2bAnchor {
    bool valid = false;
    double line0loc = 0.0;
    double first_hsync_loc = 0.0;
    double first_hsync_line = 0.0;
    double next_field = 0.0;
    int first_field = -1;
    int progressive_field = 0;
    double prev_hsync_diff = -1.0;
};

static bool inrange_host(double value, double lo, double hi) {
    return value >= lo && value <= hi;
}

static double round_nearest_line_loc_host(double line_number) {
    return std::round(0.5 * std::round(line_number / 0.5) * 10.0) / 10.0;
}

struct HostLineTimings {
    double hsync_min = 0.0;
    double hsync_max = 0.0;
    double eq_min = 0.0;
    double eq_max = 0.0;
    double vsync_min = 0.0;
    double vsync_max = 0.0;
    double hsync_nominal = 0.0;
};

static HostLineTimings host_get_line_timings(const int* pulse_lengths,
                                             int pulse_count,
                                             const VideoFormat& fmt) {
    HostLineTimings lt;
    double tol = fmt.sample_rate * 0.5e-6;
    double generous_hsync_min = fmt.hsync_width - 3.5 * tol;
    double generous_hsync_max = fmt.hsync_width + 4.0 * tol;
    double hsync_sum = 0.0;
    int hsync_count = 0;
    int npc = std::max(0, std::min(pulse_count, MAX_PULSES));
    for (int i = 0; i < npc; i++) {
        double len = (double)pulse_lengths[i];
        if (len >= generous_hsync_min && len <= generous_hsync_max) {
            hsync_sum += len;
            hsync_count++;
        }
    }
    double hsync_median_est = hsync_count > 0 ? (hsync_sum / hsync_count) : fmt.hsync_width;
    double hsync_offset = hsync_median_est - fmt.hsync_width;
    lt.hsync_nominal = fmt.hsync_width + hsync_offset;
    lt.hsync_min = lt.hsync_nominal - tol;
    lt.hsync_max = lt.hsync_nominal + tol;
    lt.eq_min = fmt.eq_pulse_width + hsync_offset - tol;
    lt.eq_max = fmt.eq_pulse_width + hsync_offset + tol;
    lt.vsync_min = (fmt.vsync_width + hsync_offset) * 0.5;
    lt.vsync_max = fmt.vsync_width + hsync_offset + 2.0 * tol;
    return lt;
}

static double host_compute_meanlinelen_from_valid_pulses(const std::vector<HostValidPulse>& valid_pulses,
                                                         double nominal_line_len) {
    int longrun_start = -1;
    int longrun_len = -1;
    int currun_start = -1;
    int currun_len = 0;
    for (int i = 0; i < (int)valid_pulses.size(); i++) {
        if (valid_pulses[(size_t)i].type != PULSE_HSYNC) {
            if (currun_start >= 0 && currun_len > longrun_len) {
                longrun_start = currun_start;
                longrun_len = currun_len;
            }
            currun_start = -1;
            currun_len = 0;
        } else if (currun_start < 0) {
            currun_start = i;
            currun_len = 0;
        } else {
            currun_len++;
        }
    }
    if (currun_start >= 0 && currun_len > longrun_len) {
        longrun_start = currun_start;
        longrun_len = currun_len;
    }

    std::vector<double> linelens;
    if (longrun_start >= 1 && longrun_len >= 1) {
        for (int i = longrun_start + 1; i < longrun_start + longrun_len; i++) {
            double linelen = (double)(valid_pulses[(size_t)i].pulse.start - valid_pulses[(size_t)i - 1].pulse.start);
            double ratio = linelen / nominal_line_len;
            if (ratio >= 0.95 && ratio <= 1.05) linelens.push_back(linelen);
        }
    }
    if (linelens.empty()) return nominal_line_len;
    double sum = 0.0;
    for (double x : linelens) sum += x;
    return sum / (double)linelens.size();
}

static bool host_run_vblank_state_machine(const std::vector<HostPulse>& raw_pulses,
                                          size_t start_idx,
                                          const HostLineTimings& lt,
                                          int num_pulses,
                                          int in_line_len,
                                          std::vector<HostValidPulse>& out) {
    bool done = false;
    double num_pulses_half = num_pulses / 2.0;
    int vsync_start = -1;
    double state_end = 0.0;
    double state_length = -1.0;

    for (size_t i = start_idx; i < raw_pulses.size() && i < start_idx + 26; i++) {
        const HostPulse& p = raw_pulses[i];
        HostValidPulse spulse;
        bool have = false;
        int state = out.empty() ? -1 : out.back().type;

        if (state == -1) {
            if (inrange_host((double)p.len, lt.hsync_min, lt.hsync_max)) {
                spulse = {PULSE_HSYNC, p, false};
                have = true;
            }
        } else if (state == PULSE_HSYNC) {
            if (inrange_host((double)p.len, lt.hsync_min, lt.hsync_max)) {
                spulse = {PULSE_HSYNC, p, false};
                have = true;
            } else if (inrange_host((double)p.len, lt.eq_min, lt.eq_max)) {
                spulse = {PULSE_EQ1, p, false};
                state_length = num_pulses_half;
                have = true;
            } else if (inrange_host((double)p.len, lt.vsync_min, lt.vsync_max)) {
                vsync_start = (int)out.size() - 1;
                spulse = {PULSE_VSYNC, p, false};
                have = true;
            }
        } else if (state == PULSE_EQ1) {
            if (inrange_host((double)p.len, lt.eq_min, lt.eq_max)) {
                spulse = {PULSE_EQ1, p, false};
                have = true;
            } else if (inrange_host((double)p.len, lt.vsync_min, lt.vsync_max)) {
                vsync_start = (int)out.size() - 1;
                spulse = {PULSE_VSYNC, p, false};
                state_length = num_pulses_half;
                have = true;
            } else if (inrange_host((double)p.len, lt.hsync_min, lt.hsync_max)) {
                spulse = {PULSE_HSYNC, p, false};
                have = true;
            }
        } else if (state == PULSE_VSYNC) {
            if (inrange_host((double)p.len, lt.eq_min, lt.eq_max)) {
                spulse = {PULSE_EQ2, p, false};
                state_length = num_pulses_half;
                have = true;
            } else if (inrange_host((double)p.len, lt.vsync_min, lt.vsync_max)) {
                spulse = {PULSE_VSYNC, p, false};
                have = true;
            } else if ((double)p.start > state_end && inrange_host((double)p.len, lt.hsync_min, lt.hsync_max)) {
                spulse = {PULSE_HSYNC, p, false};
                have = true;
            }
        } else if (state == PULSE_EQ2) {
            if (inrange_host((double)p.len, lt.eq_min, lt.eq_max)) {
                spulse = {PULSE_EQ2, p, false};
                have = true;
            } else if (inrange_host((double)p.len, lt.hsync_min, lt.hsync_max)) {
                spulse = {PULSE_HSYNC, p, false};
                have = true;
                done = true;
            }
        }

        if (have && spulse.type != state) {
            if ((double)spulse.pulse.start < state_end) {
                have = false;
            } else if (state_length > 0.0) {
                state_end = (double)spulse.pulse.start + ((state_length - 0.1) * (double)in_line_len);
                state_length = -1.0;
            }
        }

        if (have) {
            bool good = out.empty() ? false :
                host_pulse_qualitycheck(out.back().type, spulse.type, spulse.pulse.start - out.back().pulse.start, in_line_len);
            spulse.good = good;
            out.push_back(spulse);
        }
        if (done && vsync_start >= 0) return true;
    }
    return done;
}

static std::vector<HostValidPulse> host_refine_pulses_exactish(const int* pulse_starts,
                                                               const int* pulse_lengths,
                                                               int pulse_count,
                                                               const VideoFormat& fmt) {
    std::vector<HostValidPulse> valid_pulses;
    int npc = std::max(0, std::min(pulse_count, MAX_PULSES));
    if (npc <= 0) return valid_pulses;
    HostLineTimings lt = host_get_line_timings(pulse_lengths, npc, fmt);
    std::vector<HostPulse> raw_pulses;
    raw_pulses.reserve((size_t)npc);
    for (int i = 0; i < npc; i++) raw_pulses.push_back({pulse_starts[i], pulse_lengths[i]});

    for (int i = 0; i < npc;) {
        HostPulse cur = raw_pulses[(size_t)i];
        if (inrange_host((double)cur.len, lt.hsync_min, lt.hsync_max)) {
            bool good = !valid_pulses.empty() &&
                host_pulse_qualitycheck(valid_pulses.back().type, PULSE_HSYNC, cur.start - valid_pulses.back().pulse.start, fmt.samples_per_line);
            valid_pulses.push_back({PULSE_HSYNC, cur, good});
            i++;
        } else if (i > 2 &&
                   inrange_host((double)cur.len, lt.eq_min, lt.eq_max) &&
                   !valid_pulses.empty() &&
                   valid_pulses.back().type == PULSE_HSYNC) {
            std::vector<HostValidPulse> seq = valid_pulses;
            bool done = host_run_vblank_state_machine(raw_pulses, (size_t)(i - 2), lt, fmt.num_eq_pulses, fmt.samples_per_line, seq);
            if (done && seq.size() > valid_pulses.size()) {
                valid_pulses = seq;
                i += (int)(seq.size() - valid_pulses.size()) + 1;
                i = std::min(i, npc);
            } else {
                i++;
            }
        } else {
            i++;
        }
    }
    return valid_pulses;
}

static HostK2bAnchor host_get_first_hsync_loc_exactish(const std::vector<HostValidPulse>& validpulses,
                                                       double meanlinelen,
                                                       const VideoFormat& fmt,
                                                       int prev_first_field,
                                                       double last_field_offset_lines,
                                                       double prev_first_hsync_loc,
                                                       double prev_hsync_diff,
                                                       int field_order_confidence) {
    HostK2bAnchor out;
    if (validpulses.empty() || meanlinelen <= 0.0) return out;

    const int field_lines[2] = {fmt.field_lines_first, fmt.field_lines_second};
    const double VSYNC_TOLERANCE_LINES = 0.5;
    const int FIRST_VBLANK_EQ_1_START = 0;
    const int FIRST_VBLANK_VSYNC_START = 1;
    const int FIRST_VBLANK_VSYNC_END = 2;
    const int FIRST_VBLANK_EQ_2_END = 3;
    const int LAST_VBLANK_EQ_1_START = 4;
    const int LAST_VBLANK_VSYNC_START = 5;
    const int LAST_VBLANK_VSYNC_END = 6;
    const int LAST_VBLANK_EQ_2_END = 7;
    double field_order_lengths[4] = {-1, -1, -1, -1};
    int vblank_pulses[8] = {-1,-1,-1,-1,-1,-1,-1,-1};
    double vblank_lines[8] = {-1,-1,-1,-1,-1,-1,-1,-1};

    int last_pulse = -1;
    int group = 0;
    int field_group = 0;
    for (int i = 0; i < (int)validpulses.size(); i++) {
        if (last_pulse != -1 && validpulses[(size_t)i].good) {
            if (group == 0 &&
                validpulses[(size_t)i].pulse.start > validpulses[0].pulse.start + field_lines[0] * meanlinelen) {
                group = 4;
                field_group = 2;
            }
            int last_type = validpulses[(size_t)last_pulse].type;
            int type = validpulses[(size_t)i].type;
            if (last_type == PULSE_HSYNC && type > PULSE_HSYNC) {
                vblank_pulses[0 + group] = validpulses[(size_t)i].pulse.start;
                field_order_lengths[0 + field_group] =
                    round_nearest_line_loc_host((validpulses[(size_t)i].pulse.start - validpulses[(size_t)last_pulse].pulse.start) / meanlinelen);
            } else if (last_type == PULSE_EQ1 && type == PULSE_VSYNC) {
                vblank_pulses[1 + group] = validpulses[(size_t)i].pulse.start;
            } else if (last_type == PULSE_VSYNC && type == PULSE_EQ2) {
                vblank_pulses[2 + group] = validpulses[(size_t)i].pulse.start;
            } else if (last_type > PULSE_HSYNC && type == PULSE_HSYNC) {
                vblank_pulses[3 + group] = validpulses[(size_t)last_pulse].pulse.start;
                field_order_lengths[1 + field_group] =
                    round_nearest_line_loc_host((validpulses[(size_t)i].pulse.start - validpulses[(size_t)last_pulse].pulse.start) / meanlinelen);
            }
        }
        last_pulse = i;
    }

    double first_field_lengths[4];
    double second_field_lengths[4];
    double progressive_field_lengths[4];
    if (fmt.system == VideoSystem::NTSC) {
        first_field_lengths[0]=1; first_field_lengths[1]=0.5; first_field_lengths[2]=0.5; first_field_lengths[3]=1;
        second_field_lengths[0]=0.5; second_field_lengths[1]=1; second_field_lengths[2]=1; second_field_lengths[3]=0.5;
        progressive_field_lengths[0]=1; progressive_field_lengths[1]=0.5; progressive_field_lengths[2]=1; progressive_field_lengths[3]=0.5;
    } else {
        first_field_lengths[0]=0.5; first_field_lengths[1]=0.5; first_field_lengths[2]=1; first_field_lengths[3]=1;
        second_field_lengths[0]=1; second_field_lengths[1]=1; second_field_lengths[2]=0.5; second_field_lengths[3]=0.5;
        progressive_field_lengths[0]=0.5; progressive_field_lengths[1]=0.5; progressive_field_lengths[2]=0.5; progressive_field_lengths[3]=0.5;
    }

    double interlaced_consensus = 0.0;
    double interlaced_detected = 0.0;
    double progressive_consensus = 0.0;
    double progressive_detected = 0.0;
    for (int i = 0; i < 4; i++) {
        double fl = field_order_lengths[i];
        if (fl == first_field_lengths[i]) { interlaced_consensus += 1.0; interlaced_detected += 1.0; }
        if (fl == second_field_lengths[i]) interlaced_detected += 1.0;
        if (fl == progressive_field_lengths[i]) { progressive_consensus += 1.0; progressive_detected += 1.0; }
        if (fl != -1) progressive_detected += 1.0;
    }

    bool first_field;
    if (prev_first_field == -1) {
        first_field = interlaced_detected == 0.0 || std::round(interlaced_consensus / std::max(1.0, interlaced_detected)) == 1.0;
    } else {
        first_field = !prev_first_field;
    }

    int first_field_conf = 0;
    int second_field_conf = 0;
    if (interlaced_detected > 0.0) {
        if (prev_first_hsync_loc < 0.0) field_order_confidence = std::min(field_order_confidence, 50);
        double weighting = interlaced_detected / 4.0;
        first_field_conf = (int)std::llround((interlaced_consensus / interlaced_detected) * weighting * 100.0);
        second_field_conf = (int)std::llround(((interlaced_detected - interlaced_consensus) / interlaced_detected) * weighting * 100.0);
        if (first_field_conf >= field_order_confidence && first_field_conf > second_field_conf) first_field = true;
        else if (second_field_conf >= field_order_confidence && second_field_conf > first_field_conf) first_field = false;
    }

    double vsync_section_lines = fmt.num_eq_pulses / 2.0;
    double current_field_lines = first_field ? field_lines[0] : field_lines[1];
    double previous_field_lines = first_field ? field_lines[1] : field_lines[0];
    double* current_field_lengths = first_field ? first_field_lengths : second_field_lengths;
    double line0loc_line = 0.0;
    double hsync_start_line;
    vblank_lines[FIRST_VBLANK_EQ_1_START] = line0loc_line + current_field_lengths[0];
    vblank_lines[FIRST_VBLANK_VSYNC_START] = vblank_lines[FIRST_VBLANK_EQ_1_START] + vsync_section_lines;
    vblank_lines[FIRST_VBLANK_VSYNC_END] = vblank_lines[FIRST_VBLANK_VSYNC_START] + vsync_section_lines;
    vblank_lines[FIRST_VBLANK_EQ_2_END] = vblank_lines[FIRST_VBLANK_VSYNC_END] + vsync_section_lines - 0.5;
    hsync_start_line = vblank_lines[FIRST_VBLANK_EQ_2_END] + current_field_lengths[1];
    vblank_lines[LAST_VBLANK_EQ_1_START] = current_field_lines + current_field_lengths[2];
    vblank_lines[LAST_VBLANK_VSYNC_START] = vblank_lines[LAST_VBLANK_EQ_1_START] + vsync_section_lines;
    vblank_lines[LAST_VBLANK_VSYNC_END] = vblank_lines[LAST_VBLANK_VSYNC_START] + vsync_section_lines;
    vblank_lines[LAST_VBLANK_EQ_2_END] = vblank_lines[LAST_VBLANK_VSYNC_END] + vsync_section_lines - 0.5;

    auto calc_sync = [&](int first_idx, int second_idx, double& acc_loc, int& acc_count, double& acc_offset) {
        int first_pulse = vblank_pulses[first_idx];
        int second_pulse = vblank_pulses[second_idx];
        if (first_pulse == -1 || second_pulse == -1 || meanlinelen == 0.0) return;
        double actual_lines = ((double)first_pulse - (double)second_pulse) / meanlinelen;
        double expected_lines = vblank_lines[first_idx] - vblank_lines[second_idx];
        if (actual_lines < expected_lines + VSYNC_TOLERANCE_LINES &&
            actual_lines > expected_lines - VSYNC_TOLERANCE_LINES) {
            acc_offset += actual_lines - expected_lines;
            acc_loc += (double)second_pulse + meanlinelen * (hsync_start_line - vblank_lines[second_idx]);
            acc_count += 1;
        }
    };

    double first_vblank_first_hsync_loc = 0.0, first_vblank_offset = 0.0;
    int first_vblank_valid_count = 0;
    int first_indexes[4] = {FIRST_VBLANK_EQ_1_START, FIRST_VBLANK_VSYNC_START, FIRST_VBLANK_VSYNC_END, FIRST_VBLANK_EQ_2_END};
    for (int a = 0; a < 4; a++) for (int b = a + 1; b < 4; b++) calc_sync(first_indexes[a], first_indexes[b], first_vblank_first_hsync_loc, first_vblank_valid_count, first_vblank_offset);

    double last_vblank_first_hsync_loc = 0.0, last_vblank_offset = 0.0;
    int last_vblank_valid_count = 0;
    int last_indexes[4] = {LAST_VBLANK_EQ_1_START, LAST_VBLANK_VSYNC_START, LAST_VBLANK_VSYNC_END, LAST_VBLANK_EQ_2_END};
    for (int a = 0; a < 4; a++) for (int b = a + 1; b < 4; b++) calc_sync(last_indexes[a], last_indexes[b], last_vblank_first_hsync_loc, last_vblank_valid_count, last_vblank_offset);

    double first_hsync_loc = 0.0;
    int valid_location_count = 0;
    double offset = 0.0;
    double first_est = first_vblank_valid_count ? first_vblank_first_hsync_loc / first_vblank_valid_count : 0.0;
    double last_est = last_vblank_valid_count ? last_vblank_first_hsync_loc / last_vblank_valid_count : 0.0;
    if (first_vblank_valid_count && last_vblank_valid_count &&
        first_est < last_est + VSYNC_TOLERANCE_LINES * meanlinelen &&
        first_est > last_est - VSYNC_TOLERANCE_LINES * meanlinelen) {
        first_hsync_loc = first_vblank_first_hsync_loc + last_vblank_first_hsync_loc;
        valid_location_count = first_vblank_valid_count + last_vblank_valid_count;
        offset = first_vblank_offset + last_vblank_offset;
        for (int a = 0; a < 4; a++) for (int b = 0; b < 4; b++) calc_sync(first_indexes[a], last_indexes[b], first_hsync_loc, valid_location_count, offset);
    } else if (first_vblank_valid_count == 6 ||
               (prev_first_hsync_loc <= 0.0 && first_vblank_valid_count > last_vblank_valid_count)) {
        first_hsync_loc = first_vblank_first_hsync_loc;
        valid_location_count = first_vblank_valid_count;
        offset = first_vblank_offset;
    } else if (last_vblank_valid_count == 6 ||
               (prev_first_hsync_loc <= 0.0 && last_vblank_valid_count > first_vblank_valid_count)) {
        first_hsync_loc = last_vblank_first_hsync_loc;
        valid_location_count = last_vblank_valid_count;
        offset = last_vblank_offset;
    }

    double estimated_hsync_field_lines = (fmt.system == VideoSystem::NTSC) ? previous_field_lines : current_field_lines;
    int estimated_hsync_loc = (int)std::llround((last_field_offset_lines + estimated_hsync_field_lines + prev_first_hsync_loc / meanlinelen) * meanlinelen);
    bool used_estimated = false;
    if (valid_location_count == 0 && prev_first_hsync_loc > 0.0) {
        double estimated_with_offset = (prev_hsync_diff <= 0.5 && prev_hsync_diff >= -0.5)
            ? estimated_hsync_loc + meanlinelen * prev_hsync_diff
            : (double)estimated_hsync_loc;
        if (estimated_with_offset <= 0.0) estimated_with_offset = validpulses.empty() ? 0.0 : (double)validpulses.front().pulse.start;
        first_hsync_loc += estimated_with_offset;
        valid_location_count += 1;
        used_estimated = true;
    }

    if (valid_location_count <= 0) return out;
    offset /= valid_location_count;
    first_hsync_loc = std::round((first_hsync_loc + offset) / valid_location_count);
    if (!used_estimated) prev_hsync_diff = (first_hsync_loc - estimated_hsync_loc) / meanlinelen;

    double hsync_offset = 0.0;
    int hsync_count = 0;
    for (const HostValidPulse& p : validpulses) {
        if (p.type != PULSE_HSYNC || !p.good) continue;
        double lineloc = ((double)p.pulse.start - first_hsync_loc) / meanlinelen + hsync_start_line;
        int rlineloc = (int)std::llround(lineloc);
        if (rlineloc > current_field_lines) break;
        if (rlineloc >= hsync_start_line) {
            hsync_offset += first_hsync_loc + meanlinelen * ((double)rlineloc - hsync_start_line) - (double)p.pulse.start;
            hsync_count++;
        }
    }
    if (hsync_count > 0) {
        hsync_offset /= hsync_count;
        first_hsync_loc -= hsync_offset;
    }

    out.valid = true;
    out.line0loc = first_hsync_loc - meanlinelen * hsync_start_line;
    out.first_hsync_loc = first_hsync_loc;
    out.first_hsync_line = hsync_start_line;
    out.next_field = first_hsync_loc + meanlinelen * (vblank_lines[LAST_VBLANK_EQ_1_START] - hsync_start_line);
    out.first_field = first_field ? 1 : 0;
    out.progressive_field = 0;
    out.prev_hsync_diff = prev_hsync_diff;
    return out;
}

static std::vector<double> host_valid_pulses_to_linelocs_exactish(const std::vector<HostValidPulse>& validpulses,
                                                                  double reference_pulse,
                                                                  double reference_line,
                                                                  double meanlinelen,
                                                                  int proclines) {
    std::vector<double> line_locations((size_t)proclines, 0.0);
    std::vector<double> pulse_starts;
    pulse_starts.reserve(validpulses.size());
    for (const HostValidPulse& p : validpulses) pulse_starts.push_back((double)p.pulse.start);
    std::sort(pulse_starts.begin(), pulse_starts.end());
    double max_allowed = meanlinelen / 1.5;
    size_t current_pulse_index = 0;
    for (int line = 0; line < proclines; line++) {
        line_locations[(size_t)line] = reference_pulse + meanlinelen * ((double)line - reference_line);
        if (current_pulse_index >= pulse_starts.size()) continue;
        double current_distance = fabs(pulse_starts[current_pulse_index] - line_locations[(size_t)line]);
        double smallest_distance = max_allowed;
        double current_pulse_sample_location = -1.0;
        size_t pulse_search_index = current_pulse_index;
        while (pulse_search_index + 1 < pulse_starts.size()) {
            if (current_distance <= smallest_distance) {
                smallest_distance = current_distance;
                current_pulse_index = pulse_search_index;
                current_pulse_sample_location = pulse_starts[pulse_search_index];
            }
            double next_distance = fabs(pulse_starts[pulse_search_index + 1] - line_locations[(size_t)line]);
            if (next_distance > current_distance) break;
            current_distance = next_distance;
            pulse_search_index++;
        }
        if (current_pulse_sample_location >= 0.0) {
            line_locations[(size_t)line] = current_pulse_sample_location;
            current_pulse_index++;
        }
    }
    return line_locations;
}

static std::vector<HostLineMarker> build_host_line_markers(const int* pulse_starts,
                                                           const int* pulse_lengths,
                                                           const int* pulse_types,
                                                           int pulse_count,
                                                           const K3Debug& dbg,
                                                           const VideoFormat& fmt,
                                                           HostMarkerMode mode = HOST_MARKERS_BASE) {
    std::vector<HostLineMarker> markers;
    int npc_clamped = std::max(0, std::min(pulse_count, MAX_PULSES));
    double hsync_nominal = fmt.hsync_width + dbg.hsync_offset;
    double tiny_hsync_max = hsync_nominal * 0.75;
    double surrogate_gap_max = fmt.samples_per_line * 0.25;
    std::vector<int> work_starts;
    std::vector<int> work_lengths;
    std::vector<int> work_types;
    work_starts.reserve((size_t)npc_clamped);
    work_lengths.reserve((size_t)npc_clamped);
    work_types.reserve((size_t)npc_clamped);

    if (mode == HOST_MARKERS_BRIDGED) {
        double bridge_gap_max = fmt.samples_per_line * 0.12;
        double bridge_span_max = fmt.samples_per_line * 0.60;
        for (int i = 0; i < npc_clamped;) {
            int start = pulse_starts[i];
            int end = pulse_starts[i] + pulse_lengths[i];
            int j = i + 1;
            while (j < npc_clamped) {
                int next_start = pulse_starts[j];
                int next_end = pulse_starts[j] + pulse_lengths[j];
                int gap = next_start - end;
                int span = next_end - start;
                if ((double)gap > bridge_gap_max || (double)span > bridge_span_max) break;
                end = std::max(end, next_end);
                j++;
            }
            int merged_len = end - start;
            work_starts.push_back(start);
            work_lengths.push_back(merged_len);
            work_types.push_back(classify_host_pulse_type((double)merged_len, dbg, fmt));
            i = j;
        }
    } else if (mode == HOST_MARKERS_K2_DEBOUNCE) {
        double nominal_h_tol = hsync_nominal * 0.18;
        double refractory_until_offset = fmt.samples_per_line * 0.85;
        for (int i = 0; i < npc_clamped;) {
            work_starts.push_back(pulse_starts[i]);
            work_lengths.push_back(pulse_lengths[i]);
            work_types.push_back(pulse_types[i]);
            bool nominal_h = (pulse_types[i] == PULSE_HSYNC &&
                              fabs((double)pulse_lengths[i] - hsync_nominal) <= nominal_h_tol);
            if (!nominal_h) {
                i++;
                continue;
            }
            int refractory_until = pulse_starts[i] + (int)refractory_until_offset;
            int j = i + 1;
            while (j < npc_clamped && pulse_starts[j] < refractory_until) {
                j++;
            }
            i = j;
        }
    } else if (mode == HOST_MARKERS_K2_KEEP_FIRST || mode == HOST_MARKERS_K2_MERGE_CLUSTER) {
        double cluster_gap_max = fmt.samples_per_line * 0.08;
        double cluster_span_max = fmt.samples_per_line * 0.35;
        for (int i = 0; i < npc_clamped;) {
            int cluster_start = pulse_starts[i];
            int cluster_end = pulse_starts[i] + pulse_lengths[i];
            int j = i + 1;
            while (j < npc_clamped) {
                int gap = pulse_starts[j] - (pulse_starts[j - 1] + pulse_lengths[j - 1]);
                int span = (pulse_starts[j] + pulse_lengths[j]) - cluster_start;
                if ((double)gap > cluster_gap_max || (double)span > cluster_span_max) break;
                cluster_end = std::max(cluster_end, pulse_starts[j] + pulse_lengths[j]);
                j++;
            }
            if (mode == HOST_MARKERS_K2_KEEP_FIRST || j == i + 1) {
                work_starts.push_back(pulse_starts[i]);
                work_lengths.push_back(pulse_lengths[i]);
                work_types.push_back(pulse_types[i]);
            } else {
                int merged_len = cluster_end - cluster_start;
                work_starts.push_back(cluster_start);
                work_lengths.push_back(merged_len);
                work_types.push_back(classify_host_pulse_type((double)merged_len, dbg, fmt));
            }
            i = j;
        }
    } else if (mode == HOST_MARKERS_K2_VALIDATED) {
        std::vector<int> host_types;
        std::vector<int> host_valid;
        host_types.reserve((size_t)npc_clamped);
        host_valid.assign((size_t)npc_clamped, 0);
        for (int i = 0; i < npc_clamped; i++) {
            int type = pulse_types[i];
            if (type != PULSE_EQ1 && type != PULSE_EQ2 && type != PULSE_VSYNC && type != PULSE_HSYNC) {
                type = classify_host_pulse_type((double)pulse_lengths[i], dbg, fmt);
            }
            host_types.push_back(type);
        }
        for (int i = 0; i < npc_clamped; i++) {
            bool valid_prev = false;
            bool valid_next = false;
            if (i > 0) {
                valid_prev = host_pulse_qualitycheck(
                    host_types[(size_t)i - 1],
                    host_types[(size_t)i],
                    pulse_starts[i] - pulse_starts[i - 1],
                    fmt.samples_per_line);
            }
            if (i + 1 < npc_clamped) {
                valid_next = host_pulse_qualitycheck(
                    host_types[(size_t)i],
                    host_types[(size_t)i + 1],
                    pulse_starts[i + 1] - pulse_starts[i],
                    fmt.samples_per_line);
            }

            double len = (double)pulse_lengths[i];
            bool nominal_h = host_types[(size_t)i] == PULSE_HSYNC &&
                             fabs(len - hsync_nominal) <= hsync_nominal * 0.18;
            bool eq_or_v = host_types[(size_t)i] == PULSE_EQ1 ||
                           host_types[(size_t)i] == PULSE_EQ2 ||
                           host_types[(size_t)i] == PULSE_VSYNC;
            bool fallback_h = host_types[(size_t)i] == PULSE_HSYNC && !nominal_h;

            bool valid = false;
            if (nominal_h) {
                valid = valid_prev || valid_next;
            } else if (eq_or_v) {
                valid = valid_prev || valid_next;
            } else if (fallback_h) {
                bool near_h_width = len >= hsync_nominal * 0.55 && len <= hsync_nominal * 1.45;
                valid = near_h_width && valid_prev && valid_next;
            }

            host_valid[(size_t)i] = valid ? 1 : 0;
        }

        for (int i = 0; i < npc_clamped; i++) {
            if (!host_valid[(size_t)i]) continue;
            work_starts.push_back(pulse_starts[i]);
            work_lengths.push_back(pulse_lengths[i]);
            work_types.push_back(host_types[(size_t)i]);
        }
    } else {
        for (int i = 0; i < npc_clamped; i++) {
            work_starts.push_back(pulse_starts[i]);
            work_lengths.push_back(pulse_lengths[i]);
            work_types.push_back(pulse_types[i]);
        }
    }

    int work_count = (int)work_starts.size();

    for (int i = 0; i < work_count; i++) {
        bool surrogate_eq = false;
        if ((work_types[i] == PULSE_EQ1 || work_types[i] == PULSE_EQ2) &&
            i + 1 < work_count &&
            work_types[i + 1] == PULSE_HSYNC &&
            (double)work_lengths[i + 1] <= tiny_hsync_max &&
            (double)(work_starts[i + 1] - work_starts[i]) <= surrogate_gap_max) {
            surrogate_eq = true;
        }

        bool suppress_tiny_hsync = false;
        if (work_types[i] == PULSE_HSYNC &&
            (double)work_lengths[i] <= tiny_hsync_max &&
            i > 0 &&
            (work_types[i - 1] == PULSE_EQ1 || work_types[i - 1] == PULSE_EQ2) &&
            (double)(work_starts[i] - work_starts[i - 1]) <= surrogate_gap_max) {
            suppress_tiny_hsync = true;
        }

        if (surrogate_eq || (work_types[i] == PULSE_HSYNC && !suppress_tiny_hsync)) {
            markers.push_back({ work_starts[i], work_lengths[i], i });
        }
    }

    std::vector<HostLineMarker> deduped;
    double duplicate_gap_max = fmt.samples_per_line * 0.75;
    for (size_t cluster_start = 0; cluster_start < markers.size();) {
        size_t best_idx = cluster_start;
        double best_score = fabs((double)markers[cluster_start].length - hsync_nominal);
        size_t cluster_end = cluster_start + 1;
        while (cluster_end < markers.size() &&
               (double)(markers[cluster_end].start - markers[cluster_start].start) < duplicate_gap_max) {
            double score = fabs((double)markers[cluster_end].length - hsync_nominal);
            if (score < best_score ||
                (score == best_score && markers[cluster_end].start < markers[best_idx].start)) {
                best_idx = cluster_end;
                best_score = score;
            }
            cluster_end++;
        }
        if (mode == HOST_MARKERS_SUPPRESS_DENSE) {
            int cluster_size = (int)(cluster_end - cluster_start);
            int nominalish_count = 0;
            for (size_t k = cluster_start; k < cluster_end; k++) {
                double len = (double)markers[k].length;
                if (classify_host_pulse_type(len, dbg, fmt) == PULSE_HSYNC &&
                    fabs(len - hsync_nominal) <= hsync_nominal * 0.15) {
                    nominalish_count++;
                }
            }
            if ((cluster_size >= 4 && nominalish_count <= 1) ||
                (cluster_size >= 3 && nominalish_count == 0)) {
                cluster_start = cluster_end;
                continue;
            }
        }
        deduped.push_back(markers[best_idx]);
        cluster_start = cluster_end;
    }
    return deduped;
}

static HostRefTrackResult host_track_linelocs_reference(const K3Debug& dbg,
                                                        const int* pulse_starts,
                                                        const int* pulse_lengths,
                                                        const int* pulse_types,
                                                        int pulse_count,
                                                        const VideoFormat& fmt,
                                                        HostMarkerMode mode = HOST_MARKERS_BASE) {
    HostRefTrackResult result;
    if (dbg.meanlinelen <= 0.0 || fmt.lines_per_frame <= 0) return result;

    std::vector<HostLineMarker> markers =
        build_host_line_markers(pulse_starts, pulse_lengths, pulse_types, pulse_count, dbg, fmt, mode);

    struct State {
        double cost = 0.0;
        int cursor = 0;
        double loc = 0.0;
        double prev_spacing = 0.0;
        int prev_state = -1;
        bool matched = false;
    };

    result.locs.assign((size_t)fmt.lines_per_frame, 0.0);
    double max_allowed_distance = dbg.meanlinelen / 1.5;
    double hsync_nominal = fmt.hsync_width + dbg.hsync_offset;
    double unmatched_penalty = fmt.samples_per_line * 0.35;

    std::vector<State> states(1);
    states[0].loc = dbg.ref_position - dbg.meanlinelen * dbg.ref_line - dbg.meanlinelen;
    states[0].prev_spacing = dbg.meanlinelen;
    states[0].cursor = 0;

    std::vector<std::vector<State>> history;
    history.reserve(fmt.lines_per_frame);

    for (int line = 0; line < fmt.lines_per_frame; line++) {
        double expected = dbg.ref_position + dbg.meanlinelen * (line - dbg.ref_line);
        std::vector<State> next_states;
        next_states.reserve(states.size() * 6);

        for (size_t si = 0; si < states.size(); si++) {
            const State& st = states[si];

            State fallback;
            fallback.loc = expected;
            fallback.cursor = st.cursor;
            fallback.prev_spacing = expected - st.loc;
            fallback.cost = st.cost
                            + fabs(expected - dbg.meanlinelen - st.loc + st.prev_spacing) * 0.15
                            + unmatched_penalty;
            fallback.prev_state = (int)si;
            fallback.matched = false;
            next_states.push_back(fallback);

            int search = st.cursor;
            int added = 0;
            while (search < (int)markers.size()) {
                double pos = (double)markers[(size_t)search].start;
                if (pos < expected - max_allowed_distance) {
                    search++;
                    continue;
                }
                if (pos > expected + max_allowed_distance) break;

                double spacing = pos - st.loc;
                double score = st.cost;
                score += fabs(pos - expected);
                score += 1.6 * fabs(spacing - dbg.meanlinelen);
                score += 0.6 * fabs(spacing - st.prev_spacing);
                score += 3.0 * fabs((double)markers[(size_t)search].length - hsync_nominal);

                State cand;
                cand.loc = pos;
                cand.cursor = search + 1;
                cand.prev_spacing = spacing;
                cand.cost = score;
                cand.prev_state = (int)si;
                cand.matched = true;
                next_states.push_back(cand);

                search++;
                added++;
                if (added >= 4) break;
            }
        }

        std::sort(next_states.begin(), next_states.end(),
                  [](const State& a, const State& b) { return a.cost < b.cost; });

        std::vector<State> pruned;
        pruned.reserve(8);
        for (const State& cand : next_states) {
            bool duplicate = false;
            for (const State& kept : pruned) {
                if (kept.cursor == cand.cursor && fabs(kept.loc - cand.loc) < 0.5) {
                    duplicate = true;
                    break;
                }
            }
            if (!duplicate) pruned.push_back(cand);
            if ((int)pruned.size() >= 8) break;
        }
        history.push_back(pruned);
        states = pruned;
        if (states.empty()) return result;
    }

    int best_idx = 0;
    for (int i = 1; i < (int)states.size(); i++) {
        if (states[i].cost < states[best_idx].cost) best_idx = i;
    }

    for (int line = fmt.lines_per_frame - 1; line >= 0; line--) {
        const State& st = history[(size_t)line][(size_t)best_idx];
        result.locs[(size_t)line] = st.loc;
        best_idx = st.prev_state;
        if (best_idx < 0 && line > 0) break;
    }

    result.valid = true;
    return result;
}

static double local_line_error(double prev_spacing, double this_spacing, double next_spacing, double nominal) {
    return fabs(prev_spacing - nominal) + fabs(this_spacing - nominal) + fabs(next_spacing - nominal);
}

static const char* width_bucket(double len, double nominal) {
    if (len < nominal * 0.6) return "tiny";
    if (len > nominal * 1.75) return "oversize";
    if (len < nominal * 0.85) return "short";
    if (len > nominal * 1.25) return "wide";
    return "normal";
}

static const char* classify_host_pulse_kind(double len, const K3Debug& dbg, const VideoFormat& fmt) {
    switch (classify_host_pulse_type(len, dbg, fmt)) {
        case PULSE_VSYNC: return "V";
        case PULSE_EQ1: return "EQ";
        case PULSE_HSYNC: {
            double hsync_nominal = fmt.hsync_width + dbg.hsync_offset;
            if (fabs(len - hsync_nominal) <= hsync_nominal * 0.15) return "H";
            return "fallback-H";
        }
        default: return "fallback-H";
    }
}

static HostK3MatchDetail replay_k3_match_for_line(const K3Debug& dbg,
                                                  const int* pulse_starts,
                                                  const int* pulse_types,
                                                  int line,
                                                  int lines_per_frame) {
    HostK3MatchDetail detail;
    if (line < 0 || line >= lines_per_frame) return detail;

    int hsync_starts[MAX_PULSES];
    int num_hsyncs = 0;
    int npc_clamped = std::max(0, std::min(dbg.npc, MAX_PULSES));
    for (int i = 0; i < npc_clamped && num_hsyncs < MAX_PULSES; i++) {
        if (pulse_types[i] == PULSE_HSYNC) {
            hsync_starts[num_hsyncs++] = pulse_starts[i];
        }
    }

    detail.valid = true;
    detail.line = line;
    detail.max_allowed_distance = dbg.meanlinelen / 1.5;

    int cur_hsync = 0;
    for (int l = 0; l <= line; l++) {
        double expected = dbg.ref_position + dbg.meanlinelen * (l - dbg.ref_line);
        int cursor_before = cur_hsync;
        double best_dist = detail.max_allowed_distance;
        int best_pos = -1;
        int best_slot = -1;
        int search = cur_hsync;

        while (search < num_hsyncs) {
            double dist = fabs((double)hsync_starts[search] - expected);
            if (dist <= best_dist) {
                best_dist = dist;
                best_pos = hsync_starts[search];
                best_slot = search;
            }
            if (search + 1 < num_hsyncs) {
                double next_dist = fabs((double)hsync_starts[search + 1] - expected);
                if (next_dist > dist) break;
            } else {
                break;
            }
            search++;
        }

        if (best_pos >= 0) {
            cur_hsync = best_slot + 1;
        }

        if (l == line) {
            detail.expected = expected;
            detail.cursor_before = cursor_before;
            detail.cursor_after = cur_hsync;
            detail.matched = (best_pos >= 0);
            detail.chosen_hsync_slot = best_slot;
            detail.chosen_pos = best_pos;
            detail.match_dist = (best_pos >= 0) ? best_dist : detail.max_allowed_distance;
            if (best_slot > 0) detail.prev_hsync = hsync_starts[best_slot - 1];
            if (best_slot >= 0 && best_slot + 1 < num_hsyncs) detail.next_hsync = hsync_starts[best_slot + 1];
            break;
        }
    }

    return detail;
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
        alloc(&d_linelocs_coarse, n * fmt.lines_per_frame * sizeof(double));
        alloc(&d_is_first_field, n * sizeof(int));
        alloc(&d_k3_bad_spacing_count, sizeof(int));
        alloc(&d_k3_isolated_spacing_count, sizeof(int));
        alloc(&d_k3_large_jump_count, sizeof(int));
        alloc(&d_k4_large_delta_count, sizeof(int));
        alloc(&d_k4_isolated_jump_count, sizeof(int));
        alloc(&d_k4_refined_sync_like_count, sizeof(int));
        alloc(&d_k5_coarse_bad_geom_line_count, sizeof(int));
        alloc(&d_k5_coarse_sync_like_pixel_count, sizeof(int));
        alloc(&d_tbc_luma,   n * tbc_field_size * sizeof(uint16_t));
        alloc(&d_tbc_chroma, n * tbc_field_size * sizeof(uint16_t));
        alloc(&d_k5_oob_pixel_count, sizeof(int));
        alloc(&d_k5_bad_geom_line_count, sizeof(int));
        alloc(&d_k5_sync_like_pixel_count, sizeof(int));
        alloc(&d_k5_sync_like_line_counts, n * fmt.output_field_lines * sizeof(int));
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
    safe_free(&d_linelocs_coarse);
    safe_free(&d_is_first_field);
    safe_free(&d_k3_bad_spacing_count);
    safe_free(&d_k3_isolated_spacing_count);
    safe_free(&d_k3_large_jump_count);
    safe_free(&d_k4_large_delta_count);
    safe_free(&d_k4_isolated_jump_count);
    safe_free(&d_k4_refined_sync_like_count);
    safe_free(&d_k5_coarse_bad_geom_line_count);
    safe_free(&d_k5_coarse_sync_like_pixel_count);
    safe_free(&d_tbc_luma);
    safe_free(&d_tbc_chroma);
    safe_free(&d_k5_oob_pixel_count);
    safe_free(&d_k5_bad_geom_line_count);
    safe_free(&d_k5_sync_like_pixel_count);
    safe_free(&d_k5_sync_like_line_counts);
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

    bool enable_k2b_host = !getenv("CUVHS_K2B_DISABLE");
    bool log_k2b_host = env_flag_enabled("CUVHS_K2B_LOG");
    if (enable_k2b_host) {
        auto* h_pulse_counts_k2b = new int[fields_loaded];
        auto* h_pulse_starts_k2b = new int[(size_t)fields_loaded * MAX_PULSES];
        auto* h_pulse_lengths_k2b = new int[(size_t)fields_loaded * MAX_PULSES];
        auto* h_out_counts_k2b = new int[fields_loaded];
        auto* h_out_starts_k2b = new int[(size_t)fields_loaded * MAX_PULSES];
        auto* h_out_lengths_k2b = new int[(size_t)fields_loaded * MAX_PULSES];
        auto* h_out_types_k2b = new int[(size_t)fields_loaded * MAX_PULSES];
        auto* h_out_linelocs_k2b = new double[(size_t)fields_loaded * fmt.lines_per_frame];
        auto* h_out_is_first_k2b = new int[fields_loaded];

        cudaMemcpy(h_pulse_counts_k2b, d_pulse_count,
                   fields_loaded * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_pulse_starts_k2b, d_pulse_starts,
                   (size_t)fields_loaded * MAX_PULSES * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_pulse_lengths_k2b, d_pulse_lengths,
                   (size_t)fields_loaded * MAX_PULSES * sizeof(int), cudaMemcpyDeviceToHost);

        int total_before = 0;
        int total_after = 0;
        int changed_fields = 0;
        for (int field = 0; field < fields_loaded; field++) {
            int before = std::max(0, std::min(h_pulse_counts_k2b[field], MAX_PULSES));
            std::vector<HostValidPulse> validpulses = host_refine_pulses_exactish(
                h_pulse_starts_k2b + (size_t)field * MAX_PULSES,
                h_pulse_lengths_k2b + (size_t)field * MAX_PULSES,
                before,
                fmt);
            double meanlinelen = host_compute_meanlinelen_from_valid_pulses(validpulses, (double)fmt.samples_per_line);
            double prev_offset_lines = 0.0;
            if (k2b_prev_first_hsync_readloc != -1 && meanlinelen > 0.0) {
                prev_offset_lines = ((double)k2b_prev_first_hsync_readloc - (double)(raw_offset + chunk_field_offsets[field])) / meanlinelen;
            }
            HostK2bAnchor anchor = host_get_first_hsync_loc_exactish(
                validpulses,
                meanlinelen,
                fmt,
                k2b_prev_first_field,
                prev_offset_lines,
                k2b_prev_first_hsync_loc,
                k2b_prev_first_hsync_diff,
                75);

            int after = std::min((int)validpulses.size(), MAX_PULSES);
            for (int i = 0; i < after; i++) {
                h_out_starts_k2b[(size_t)field * MAX_PULSES + i] = validpulses[(size_t)i].pulse.start;
                h_out_lengths_k2b[(size_t)field * MAX_PULSES + i] = validpulses[(size_t)i].pulse.len;
                h_out_types_k2b[(size_t)field * MAX_PULSES + i] = validpulses[(size_t)i].type;
            }
            for (int i = after; i < MAX_PULSES; i++) {
                h_out_starts_k2b[(size_t)field * MAX_PULSES + i] = 0;
                h_out_lengths_k2b[(size_t)field * MAX_PULSES + i] = 0;
                h_out_types_k2b[(size_t)field * MAX_PULSES + i] = PULSE_HSYNC;
            }
            h_out_counts_k2b[field] = after;
            std::vector<double> locs;
            if (anchor.valid) {
                locs = host_valid_pulses_to_linelocs_exactish(
                    validpulses,
                    anchor.first_hsync_loc,
                    anchor.first_hsync_line,
                    meanlinelen,
                    fmt.lines_per_frame);
                h_out_is_first_k2b[field] = anchor.first_field;
                k2b_prev_first_field = anchor.first_field;
                k2b_prev_progressive_field = anchor.progressive_field;
                k2b_prev_first_hsync_readloc = (long long)(raw_offset + chunk_field_offsets[field]);
                k2b_prev_first_hsync_loc = anchor.first_hsync_loc;
                k2b_prev_first_hsync_diff = anchor.prev_hsync_diff;
            } else {
                locs.resize((size_t)fmt.lines_per_frame);
                double ref = before > 0 ? (double)h_pulse_starts_k2b[(size_t)field * MAX_PULSES] : 0.0;
                for (int line = 0; line < fmt.lines_per_frame; line++) {
                    locs[(size_t)line] = ref + meanlinelen * (double)line;
                }
                h_out_is_first_k2b[field] = (k2b_prev_first_field >= 0) ? (1 - k2b_prev_first_field) : 1;
            }

            std::copy(locs.begin(), locs.end(), h_out_linelocs_k2b + (size_t)field * fmt.lines_per_frame);
            total_before += before;
            total_after += after;
            if (after != before) changed_fields++;
        }

        cudaMemcpy(d_pulse_count, h_out_counts_k2b,
                   fields_loaded * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_pulse_starts, h_out_starts_k2b,
                   (size_t)fields_loaded * MAX_PULSES * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_pulse_lengths, h_out_lengths_k2b,
                   (size_t)fields_loaded * MAX_PULSES * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_pulse_types, h_out_types_k2b,
                   (size_t)fields_loaded * MAX_PULSES * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_linelocs, h_out_linelocs_k2b,
                   (size_t)fields_loaded * fmt.lines_per_frame * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_is_first_field, h_out_is_first_k2b,
                   fields_loaded * sizeof(int), cudaMemcpyHostToDevice);

        if (log_k2b_host) {
            fprintf(stderr, "  [K2b host] chunk raw_offset=%zu: pulses %d -> %d, changed_fields=%d/%d\n",
                    raw_offset, total_before, total_after, changed_fields, fields_loaded);
        }

        delete[] h_out_is_first_k2b;
        delete[] h_out_linelocs_k2b;
        delete[] h_out_types_k2b;
        delete[] h_out_lengths_k2b;
        delete[] h_out_starts_k2b;
        delete[] h_out_counts_k2b;
        delete[] h_pulse_lengths_k2b;
        delete[] h_pulse_starts_k2b;
        delete[] h_pulse_counts_k2b;
    }

    bool debug_k3_host_ref = env_flag_enabled("CUVHS_DEBUG_K3_HOST_REF");
    bool debug_k3_host_patch = env_flag_enabled("CUVHS_DEBUG_K3_HOST_PATCH");
    bool debug_k3_host_bridge = env_flag_enabled("CUVHS_DEBUG_K3_HOST_BRIDGE");
    bool debug_k3_host_swarm = env_flag_enabled("CUVHS_DEBUG_K3_HOST_SWARM");
    bool debug_k2_host_debounce = env_flag_enabled("CUVHS_DEBUG_K2_HOST_DEBOUNCE");
    bool debug_k2_host_keep_first = env_flag_enabled("CUVHS_DEBUG_K2_HOST_KEEP_FIRST");
    bool debug_k2_host_merge_cluster = env_flag_enabled("CUVHS_DEBUG_K2_HOST_MERGE_CLUSTER");
    bool debug_k2_host_validated = env_flag_enabled("CUVHS_DEBUG_K2_HOST_VALIDATED");
    HostMarkerMode host_ref_mode = HOST_MARKERS_BASE;
    const char* host_ref_mode_name = "base";
    if (debug_k2_host_validated) {
        host_ref_mode = HOST_MARKERS_K2_VALIDATED;
        host_ref_mode_name = "k2-validated";
    } else if (debug_k2_host_debounce) {
        host_ref_mode = HOST_MARKERS_K2_DEBOUNCE;
        host_ref_mode_name = "k2-debounce";
    } else if (debug_k2_host_keep_first) {
        host_ref_mode = HOST_MARKERS_K2_KEEP_FIRST;
        host_ref_mode_name = "k2-keep-first";
    } else if (debug_k2_host_merge_cluster) {
        host_ref_mode = HOST_MARKERS_K2_MERGE_CLUSTER;
        host_ref_mode_name = "k2-merge-cluster";
    } else if (debug_k3_host_bridge) {
        host_ref_mode = HOST_MARKERS_BRIDGED;
        host_ref_mode_name = "bridge";
    } else if (debug_k3_host_swarm) {
        host_ref_mode = HOST_MARKERS_SUPPRESS_DENSE;
        host_ref_mode_name = "swarm";
    }

    // K3: Classify pulses + line locations (now with correct field alignment)
    K3Debug* d_k3_debug = nullptr;
    {
        static auto dump_fields = get_dump_fields();
        if (!enable_k2b_host && (!dump_fields.empty() || debug_k3_host_ref || debug_k3_host_patch)) {
            cudaMalloc(&d_k3_debug, fields_loaded * sizeof(K3Debug));
            cudaMemset(d_k3_debug, 0, fields_loaded * sizeof(K3Debug));
        }
    }
    if (!enable_k2b_host) {
        line_locs(static_cast<int*>(d_pulse_starts),
                  static_cast<int*>(d_pulse_lengths),
                  static_cast<int*>(d_pulse_count),
                  static_cast<int*>(d_pulse_types),
                  static_cast<double*>(d_linelocs),
                  static_cast<int*>(d_is_first_field),
                  fields_loaded, fmt, d_k3_debug);
    }

    bool debug_k3_log = env_flag_enabled("CUVHS_DEBUG_K3_LOG");
    if (debug_k3_log) {
        cudaMemset(d_k3_bad_spacing_count, 0, sizeof(int));
        cudaMemset(d_k3_isolated_spacing_count, 0, sizeof(int));
        cudaMemset(d_k3_large_jump_count, 0, sizeof(int));
        line_locs_debug_analyze(static_cast<double*>(d_linelocs),
                                static_cast<int*>(d_k3_bad_spacing_count),
                                static_cast<int*>(d_k3_isolated_spacing_count),
                                static_cast<int*>(d_k3_large_jump_count),
                                fields_loaded, fmt);
        int h_k3_bad_spacing = 0;
        int h_k3_isolated_spacing = 0;
        int h_k3_large_jump = 0;
        cudaMemcpy(&h_k3_bad_spacing, d_k3_bad_spacing_count, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_k3_isolated_spacing, d_k3_isolated_spacing_count, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_k3_large_jump, d_k3_large_jump_count, sizeof(int), cudaMemcpyDeviceToHost);
        fprintf(stderr, "  [K3 spacing] chunk raw_offset=%zu: bad_spacing=%d isolated_spacing=%d large_jump=%d\n",
                raw_offset, h_k3_bad_spacing, h_k3_isolated_spacing, h_k3_large_jump);
    }

    K3Debug* h_k3_debug = nullptr;
    // Debug: dump K3 decisions
    if (d_k3_debug) {
        static auto dump_fields = get_dump_fields();
        h_k3_debug = new K3Debug[fields_loaded];
        cudaMemcpy(h_k3_debug, d_k3_debug, fields_loaded * sizeof(K3Debug), cudaMemcpyDeviceToHost);
        for (int df : dump_fields) {
            int local_idx = df;
            if (local_idx >= 0 && local_idx < fields_loaded) {
                const K3Debug& d = h_k3_debug[local_idx];
                fprintf(stderr, "  [K3 debug] field %d: npc=%d ref_idx=%d ref_pos=%.1f ref_line=%.1f "
                        "meanll=%.2f best_run=%d num_hsyncs=%d hsync_off=%.2f state=%d parity=%d "
                        "bad=%d iso=%d jump=%d minsp=%.1f maxsp=%.1f bad_lines=%d,%d,%d,%d "
                        "worst_match=%d(%.1f),%d(%.1f),%d(%.1f),%d(%.1f)\n",
                        df, d.npc, d.ref_pulse_idx, d.ref_position, d.ref_line,
                        d.meanlinelen, d.best_run_len, d.num_hsyncs, d.hsync_offset,
                        d.final_state, d.field_parity,
                        d.bad_spacing_count, d.isolated_spacing_count, d.large_jump_count,
                        d.min_spacing, d.max_spacing,
                        d.bad_lines[0], d.bad_lines[1], d.bad_lines[2], d.bad_lines[3],
                        d.worst_match_lines[0], d.worst_match_dist[0],
                        d.worst_match_lines[1], d.worst_match_dist[1],
                        d.worst_match_lines[2], d.worst_match_dist[2],
                        d.worst_match_lines[3], d.worst_match_dist[3]);
            }
        }
    }

    if (debug_k3_host_patch && h_k3_debug) {
        auto* h_linelocs_patch = new double[(size_t)fields_loaded * fmt.lines_per_frame];
        auto* h_pulse_counts_patch = new int[fields_loaded];
        auto* h_pulse_starts_patch = new int[(size_t)fields_loaded * MAX_PULSES];
        auto* h_pulse_lengths_patch = new int[(size_t)fields_loaded * MAX_PULSES];
        auto* h_pulse_types_patch = new int[(size_t)fields_loaded * MAX_PULSES];
        cudaMemcpy(h_linelocs_patch, d_linelocs,
                   (size_t)fields_loaded * fmt.lines_per_frame * sizeof(double),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(h_pulse_counts_patch, d_pulse_count,
                   fields_loaded * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_pulse_starts_patch, d_pulse_starts,
                   (size_t)fields_loaded * MAX_PULSES * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_pulse_lengths_patch, d_pulse_lengths,
                   (size_t)fields_loaded * MAX_PULSES * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_pulse_types_patch, d_pulse_types,
                   (size_t)fields_loaded * MAX_PULSES * sizeof(int), cudaMemcpyDeviceToHost);

        int applied_patches = 0;
        fprintf(stderr, "  [K3 host patch apply] chunk raw_offset=%zu:", raw_offset);
        bool printed_patch = false;
        for (int field = 0; field < fields_loaded; field++) {
            HostRefTrackResult ref = host_track_linelocs_reference(
                h_k3_debug[field],
                h_pulse_starts_patch + (size_t)field * MAX_PULSES,
                h_pulse_lengths_patch + (size_t)field * MAX_PULSES,
                h_pulse_types_patch + (size_t)field * MAX_PULSES,
                h_pulse_counts_patch[field],
                fmt);
            if (!ref.valid || ref.locs.size() < (size_t)fmt.lines_per_frame) continue;

            double* coarse_locs = h_linelocs_patch + (size_t)field * fmt.lines_per_frame;
            std::vector<HostRefPatch> field_candidates;
            for (int ll_line = 1; ll_line + 2 < fmt.lines_per_frame; ll_line++) {
                if (ll_line < fmt.active_line_start || ll_line >= fmt.active_line_start + fmt.output_field_lines) continue;
                double coarse_prev = coarse_locs[ll_line] - coarse_locs[ll_line - 1];
                double coarse_this = coarse_locs[ll_line + 1] - coarse_locs[ll_line];
                double coarse_next = coarse_locs[ll_line + 2] - coarse_locs[ll_line + 1];
                double ref_prev = ref.locs[(size_t)ll_line] - ref.locs[(size_t)ll_line - 1];
                double ref_this = ref.locs[(size_t)ll_line + 1] - ref.locs[(size_t)ll_line];
                double ref_next = ref.locs[(size_t)ll_line + 2] - ref.locs[(size_t)ll_line + 1];
                double coarse_error = local_line_error(
                    coarse_prev, coarse_this, coarse_next, (double)fmt.samples_per_line);
                double ref_error = local_line_error(
                    ref_prev, ref_this, ref_next, (double)fmt.samples_per_line);
                double improvement = coarse_error - ref_error;
                double delta = ref.locs[(size_t)ll_line] - coarse_locs[ll_line];
                bool sane_ref =
                    ref_prev > fmt.samples_per_line * 0.9 &&
                    ref_prev < fmt.samples_per_line * 1.1 &&
                    ref_this > fmt.samples_per_line * 0.9 &&
                    ref_this < fmt.samples_per_line * 1.1 &&
                    ref_next > fmt.samples_per_line * 0.9 &&
                    ref_next < fmt.samples_per_line * 1.1;
                if (!sane_ref) continue;
                if (improvement < 220.0 || std::abs(delta) < 40.0 || std::abs(delta) > 900.0) continue;
                field_candidates.push_back(
                    HostRefPatch{field, ll_line, coarse_locs[ll_line], ref.locs[(size_t)ll_line], delta, improvement});
            }

            std::sort(field_candidates.begin(), field_candidates.end(),
                      [](const HostRefPatch& a, const HostRefPatch& b) {
                          if (a.improvement != b.improvement) return a.improvement > b.improvement;
                          return a.line < b.line;
                      });

            int field_applied = 0;
            std::vector<int> patched_lines;
            for (const HostRefPatch& patch : field_candidates) {
                bool near_existing = false;
                for (int seen_line : patched_lines) {
                    if (std::abs(seen_line - patch.line) <= 2) {
                        near_existing = true;
                        break;
                    }
                }
                if (near_existing) continue;
                coarse_locs[patch.line] = patch.ref_loc;
                patched_lines.push_back(patch.line);
                applied_patches++;
                field_applied++;
                if (applied_patches <= 12) {
                    fprintf(stderr, " f%d:l%d(%+.1f)", patch.field, patch.line, patch.delta);
                    printed_patch = true;
                }
                if (field_applied >= 4 || applied_patches >= 32) break;
            }
            if (applied_patches >= 32) break;
        }
        if (!printed_patch) fprintf(stderr, " none");
        fprintf(stderr, " total=%d\n", applied_patches);
        cudaMemcpy(d_linelocs, h_linelocs_patch,
                   (size_t)fields_loaded * fmt.lines_per_frame * sizeof(double),
                   cudaMemcpyHostToDevice);
        delete[] h_pulse_types_patch;
        delete[] h_pulse_lengths_patch;
        delete[] h_pulse_starts_patch;
        delete[] h_pulse_counts_patch;
        delete[] h_linelocs_patch;
    }

    // K4: Refine Line Locations via Hsync Zero-Crossing
    bool debug_k4_log = env_flag_enabled("CUVHS_DEBUG_K4_LOG");
    cudaMemcpy(d_linelocs_coarse, d_linelocs,
               (size_t)fields_loaded * fmt.lines_per_frame * sizeof(double),
               cudaMemcpyDeviceToDevice);
    hsync_refine(static_cast<double*>(d_demod_05),
                 static_cast<double*>(d_linelocs),
                 fields_loaded,
                 fields_loaded * (int)spf_padded,
                 fmt);
    if (debug_k4_log) {
        cudaMemset(d_k4_large_delta_count, 0, sizeof(int));
        cudaMemset(d_k4_isolated_jump_count, 0, sizeof(int));
        cudaMemset(d_k4_refined_sync_like_count, 0, sizeof(int));
        hsync_refine_debug_analyze(static_cast<double*>(d_demod_05),
                                   static_cast<double*>(d_linelocs_coarse),
                                   static_cast<double*>(d_linelocs),
                                   static_cast<int*>(d_k4_large_delta_count),
                                   static_cast<int*>(d_k4_isolated_jump_count),
                                   static_cast<int*>(d_k4_refined_sync_like_count),
                                   fields_loaded,
                                   fields_loaded * (int)spf_padded,
                                   fmt);
        int h_k4_large_delta = 0;
        int h_k4_isolated = 0;
        int h_k4_sync_like = 0;
        cudaMemcpy(&h_k4_large_delta, d_k4_large_delta_count, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_k4_isolated, d_k4_isolated_jump_count, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_k4_sync_like, d_k4_refined_sync_like_count, sizeof(int), cudaMemcpyDeviceToHost);
        fprintf(stderr, "  [K4 debug] chunk raw_offset=%zu: large_delta=%d isolated_jump=%d refined_sync_like=%d\n",
                raw_offset, h_k4_large_delta, h_k4_isolated, h_k4_sync_like);
    }

    // K5: TBC Resample
    bool debug_k5_log = env_flag_enabled("CUVHS_DEBUG_K5_LOG");
    bool debug_k5_mark_oob = env_flag_enabled("CUVHS_DEBUG_K5_MARK_OOB_WHITE");
    bool debug_k5_mark_sync = env_flag_enabled("CUVHS_DEBUG_K5_MARK_SYNC_WHITE");
    bool debug_k5_compare_coarse = env_flag_enabled("CUVHS_DEBUG_K5_COMPARE_COARSE");
    bool debug_k5_lines = env_flag_enabled("CUVHS_DEBUG_K5_LINES");
    bool skip_k4 = env_flag_enabled("CUVHS_DEBUG_K4_SKIP");
    const double* k5_linelocs = skip_k4 ? static_cast<double*>(d_linelocs_coarse)
                                        : static_cast<double*>(d_linelocs);
    cudaMemset(d_k5_oob_pixel_count, 0, sizeof(int));
    cudaMemset(d_k5_bad_geom_line_count, 0, sizeof(int));
    cudaMemset(d_k5_sync_like_pixel_count, 0, sizeof(int));
    cudaMemset(d_k5_sync_like_line_counts, 0,
               (size_t)fields_loaded * fmt.output_field_lines * sizeof(int));
    tbc_resample(static_cast<double*>(d_demod),
                 k5_linelocs,
                 static_cast<uint16_t*>(d_tbc_luma),
                 static_cast<int*>(d_k5_oob_pixel_count),
                 static_cast<int*>(d_k5_bad_geom_line_count),
                 static_cast<int*>(d_k5_sync_like_pixel_count),
                 static_cast<int*>(d_k5_sync_like_line_counts),
                 debug_k5_mark_oob,
                 debug_k5_mark_sync,
                 fields_loaded,
                 fields_loaded * (int)spf_padded,
                 fmt);

    if (debug_k5_log) {
        int h_k5_oob_pixels = 0;
        int h_k5_bad_geom_lines = 0;
        int h_k5_sync_like_pixels = 0;
        cudaMemcpy(&h_k5_oob_pixels, d_k5_oob_pixel_count, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_k5_bad_geom_lines, d_k5_bad_geom_line_count, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_k5_sync_like_pixels, d_k5_sync_like_pixel_count, sizeof(int), cudaMemcpyDeviceToHost);
        fprintf(stderr, "  [K5 debug] chunk raw_offset=%zu: mode=%s oob_pixels=%d bad_geom_lines=%d sync_like_pixels=%d\n",
                raw_offset, skip_k4 ? "coarse" : "refined",
                h_k5_oob_pixels, h_k5_bad_geom_lines, h_k5_sync_like_pixels);
        if (debug_k5_lines) {
            static auto dump_fields = get_dump_fields();
            auto* h_line_counts = new int[(size_t)fields_loaded * fmt.output_field_lines];
            cudaMemcpy(h_line_counts, d_k5_sync_like_line_counts,
                       (size_t)fields_loaded * fmt.output_field_lines * sizeof(int),
                       cudaMemcpyDeviceToHost);
            for (int df : dump_fields) {
                int local_idx = df;
                if (local_idx >= 0 && local_idx < fields_loaded) {
                    const int* counts = h_line_counts + (size_t)local_idx * fmt.output_field_lines;
                    int printed = 0;
                    fprintf(stderr, "    [K5 lines] field %d:", df);
                    for (int line = 0; line < fmt.output_field_lines && printed < 8; line++) {
                        if (counts[line] > 0) {
                            fprintf(stderr, " %d(%d)", line, counts[line]);
                            printed++;
                        }
                    }
                    if (printed == 0) fprintf(stderr, " none");
                    fprintf(stderr, "\n");
                }
            }
            struct K5LineHit {
                int field;
                int line;
                int count;
            };
            K5LineHit top_hits[8];
            for (int i = 0; i < 8; i++) top_hits[i] = { -1, -1, 0 };
            int top_start = 16;
            int top_end = fmt.output_field_lines - 16;
            for (int field = 0; field < fields_loaded; field++) {
                const int* counts = h_line_counts + (size_t)field * fmt.output_field_lines;
                for (int line = top_start; line < top_end; line++) {
                    int count = counts[line];
                    if (count <= 0) continue;
                    for (int slot = 0; slot < 8; slot++) {
                        if (count > top_hits[slot].count) {
                            for (int move = 7; move > slot; move--) top_hits[move] = top_hits[move - 1];
                            top_hits[slot] = { field, line, count };
                            break;
                        }
                    }
                }
            }
            fprintf(stderr, "    [K5 top lines] chunk raw_offset=%zu:", raw_offset);
            bool any_top_hits = false;
            for (int i = 0; i < 8; i++) {
                if (top_hits[i].count > 0) {
                    fprintf(stderr, " f%d:l%d(%d)", top_hits[i].field, top_hits[i].line, top_hits[i].count);
                    any_top_hits = true;
                }
            }
            if (!any_top_hits) fprintf(stderr, " none");
            fprintf(stderr, "\n");

            auto* h_linelocs_dbg = new double[(size_t)fields_loaded * fmt.lines_per_frame];
            auto* h_linelocs_coarse_dbg = new double[(size_t)fields_loaded * fmt.lines_per_frame];
            cudaMemcpy(h_linelocs_dbg, k5_linelocs,
                       (size_t)fields_loaded * fmt.lines_per_frame * sizeof(double),
                       cudaMemcpyDeviceToHost);
            cudaMemcpy(h_linelocs_coarse_dbg, d_linelocs_coarse,
                       (size_t)fields_loaded * fmt.lines_per_frame * sizeof(double),
                       cudaMemcpyDeviceToHost);
            int* h_pulse_counts_dbg = nullptr;
            int* h_pulse_starts_dbg = nullptr;
            int* h_pulse_lengths_dbg = nullptr;
            int* h_pulse_types_dbg = nullptr;
            std::vector<HostRefTrackResult> host_ref_tracks;
            std::vector<HostRefImprovement> host_ref_improvements;
            std::vector<HostRefPatch> host_ref_patches;
            std::vector<HostRefDisagreement> host_ref_disagreements;
            if (h_k3_debug) {
                h_pulse_counts_dbg = new int[fields_loaded];
                h_pulse_starts_dbg = new int[(size_t)fields_loaded * MAX_PULSES];
                h_pulse_lengths_dbg = new int[(size_t)fields_loaded * MAX_PULSES];
                h_pulse_types_dbg = new int[(size_t)fields_loaded * MAX_PULSES];
                cudaMemcpy(h_pulse_counts_dbg, d_pulse_count,
                           fields_loaded * sizeof(int), cudaMemcpyDeviceToHost);
                cudaMemcpy(h_pulse_starts_dbg, d_pulse_starts,
                           (size_t)fields_loaded * MAX_PULSES * sizeof(int), cudaMemcpyDeviceToHost);
                cudaMemcpy(h_pulse_lengths_dbg, d_pulse_lengths,
                           (size_t)fields_loaded * MAX_PULSES * sizeof(int), cudaMemcpyDeviceToHost);
                cudaMemcpy(h_pulse_types_dbg, d_pulse_types,
                           (size_t)fields_loaded * MAX_PULSES * sizeof(int), cudaMemcpyDeviceToHost);
                if (debug_k3_host_ref) {
                    host_ref_tracks.resize((size_t)fields_loaded);
                    for (int field = 0; field < fields_loaded; field++) {
                        host_ref_tracks[(size_t)field] = host_track_linelocs_reference(
                            h_k3_debug[field],
                            h_pulse_starts_dbg + (size_t)field * MAX_PULSES,
                            h_pulse_lengths_dbg + (size_t)field * MAX_PULSES,
                            h_pulse_types_dbg + (size_t)field * MAX_PULSES,
                            h_pulse_counts_dbg[field],
                            fmt,
                            host_ref_mode);
                    }
                    for (int field = 0; field < fields_loaded; field++) {
                        const HostRefTrackResult& ref = host_ref_tracks[(size_t)field];
                        if (!ref.valid || ref.locs.size() < (size_t)fmt.lines_per_frame) continue;
                        const double* coarse_locs = h_linelocs_coarse_dbg + (size_t)field * fmt.lines_per_frame;
                        const double* ref_locs = ref.locs.data();
                        for (int ll_line = 1; ll_line + 2 < fmt.lines_per_frame; ll_line++) {
                            double coarse_prev = coarse_locs[ll_line] - coarse_locs[ll_line - 1];
                            double coarse_this = coarse_locs[ll_line + 1] - coarse_locs[ll_line];
                            double coarse_next = coarse_locs[ll_line + 2] - coarse_locs[ll_line + 1];
                            double ref_prev = ref_locs[ll_line] - ref_locs[ll_line - 1];
                            double ref_this = ref_locs[ll_line + 1] - ref_locs[ll_line];
                            double ref_next = ref_locs[ll_line + 2] - ref_locs[ll_line + 1];
                            double coarse_error = local_line_error(
                                coarse_prev, coarse_this, coarse_next, (double)fmt.samples_per_line);
                            double ref_error = local_line_error(
                                ref_prev, ref_this, ref_next, (double)fmt.samples_per_line);
                            double improvement = coarse_error - ref_error;
                            if (improvement < 60.0) continue;
                            HostRefImprovement hit{field, ll_line, coarse_error, ref_error, improvement};
                            host_ref_improvements.push_back(hit);
                            double coarse_loc = coarse_locs[ll_line];
                            double ref_loc = ref_locs[ll_line];
                            double delta = ref_loc - coarse_loc;
                            if (ll_line >= fmt.active_line_start &&
                                ll_line < fmt.active_line_start + fmt.output_field_lines &&
                                std::abs(delta) >= 40.0) {
                                host_ref_disagreements.push_back(
                                    HostRefDisagreement{
                                        field, ll_line, coarse_loc, ref_loc, delta,
                                        coarse_error, ref_error, improvement
                                    });
                            }
                            if (std::abs(delta) >= 20.0) {
                                host_ref_patches.push_back(
                                    HostRefPatch{field, ll_line, coarse_loc, ref_loc, delta, improvement});
                            }
                        }
                    }
                    std::sort(host_ref_improvements.begin(), host_ref_improvements.end(),
                              [](const HostRefImprovement& a, const HostRefImprovement& b) {
                                  if (a.improvement != b.improvement) return a.improvement > b.improvement;
                                  if (a.field != b.field) return a.field < b.field;
                                  return a.line < b.line;
                              });
                    fprintf(stderr, "    [K3 host best %s] chunk raw_offset=%zu:", host_ref_mode_name, raw_offset);
                    bool any_host_best = false;
                    int printed = 0;
                    std::vector<HostRefImprovement> printed_hits;
                    for (const HostRefImprovement& hit : host_ref_improvements) {
                        bool duplicate = false;
                        for (const HostRefImprovement& seen : printed_hits) {
                            if (seen.field == hit.field && std::abs(seen.line - hit.line) <= 1) {
                                duplicate = true;
                                break;
                            }
                        }
                        if (duplicate) continue;
                        fprintf(stderr, " f%d:l%d(+%.1f)", hit.field, hit.line, hit.improvement);
                        any_host_best = true;
                        printed_hits.push_back(hit);
                        printed++;
                        if (printed >= 8) break;
                    }
                    if (!any_host_best) fprintf(stderr, " none");
                    fprintf(stderr, "\n");

                    std::sort(host_ref_patches.begin(), host_ref_patches.end(),
                              [](const HostRefPatch& a, const HostRefPatch& b) {
                                  if (a.improvement != b.improvement) return a.improvement > b.improvement;
                                  if (a.field != b.field) return a.field < b.field;
                                  return a.line < b.line;
                              });
                    fprintf(stderr, "    [K3 host patch %s] chunk raw_offset=%zu:", host_ref_mode_name, raw_offset);
                    bool any_host_patch = false;
                    int printed_patch = 0;
                    std::vector<HostRefPatch> printed_patches;
                    for (const HostRefPatch& patch : host_ref_patches) {
                        bool duplicate = false;
                        for (const HostRefPatch& seen : printed_patches) {
                            if (seen.field == patch.field && std::abs(seen.line - patch.line) <= 1) {
                                duplicate = true;
                                break;
                            }
                        }
                        if (duplicate) continue;
                        fprintf(stderr, " f%d:l%d(%.1f->%.1f d=%+.1f)",
                                patch.field, patch.line,
                                patch.coarse_loc, patch.ref_loc, patch.delta);
                        any_host_patch = true;
                        printed_patches.push_back(patch);
                        printed_patch++;
                        if (printed_patch >= 8) break;
                    }
                    if (!any_host_patch) fprintf(stderr, " none");
                    fprintf(stderr, "\n");

                    std::sort(host_ref_disagreements.begin(), host_ref_disagreements.end(),
                              [](const HostRefDisagreement& a, const HostRefDisagreement& b) {
                                  if (a.improvement != b.improvement) return a.improvement > b.improvement;
                                  if (std::abs(a.delta) != std::abs(b.delta)) {
                                      return std::abs(a.delta) > std::abs(b.delta);
                                  }
                                  if (a.field != b.field) return a.field < b.field;
                                  return a.line < b.line;
                              });
                    fprintf(stderr, "    [K3 host disagree %s] chunk raw_offset=%zu:", host_ref_mode_name, raw_offset);
                    bool any_host_disagree = false;
                    int printed_disagree = 0;
                    std::vector<HostRefDisagreement> printed_disagreements;
                    for (const HostRefDisagreement& hit : host_ref_disagreements) {
                        bool duplicate = false;
                        for (const HostRefDisagreement& seen : printed_disagreements) {
                            if (seen.field == hit.field && std::abs(seen.line - hit.line) <= 1) {
                                duplicate = true;
                                break;
                            }
                        }
                        if (duplicate) continue;
                        fprintf(stderr, " f%d:l%d(d=%+.1f imp=%.1f)",
                                hit.field, hit.line, hit.delta, hit.improvement);
                        any_host_disagree = true;
                        printed_disagreements.push_back(hit);
                        printed_disagree++;
                        if (printed_disagree >= 8) break;
                    }
                    if (!any_host_disagree) fprintf(stderr, " none");
                    fprintf(stderr, "\n");
                }
            }
            for (int i = 0; debug_k3_host_ref && i < 4 && i < (int)host_ref_disagreements.size(); i++) {
                const HostRefDisagreement& hit = host_ref_disagreements[(size_t)i];
                int field = hit.field;
                int ll_line = hit.line;
                fprintf(stderr,
                        "    [K3 host disagree detail] f%d ll=%d coarse=%.1f ref=%.1f delta=%+.1f err=%.1f->%.1f\n",
                        field, ll_line, hit.coarse_loc, hit.ref_loc, hit.delta,
                        hit.coarse_error, hit.ref_error);
                if (h_k3_debug && h_pulse_counts_dbg && h_pulse_starts_dbg &&
                    h_pulse_lengths_dbg && h_pulse_types_dbg &&
                    field >= 0 && field < fields_loaded) {
                    const int* field_pulse_starts = h_pulse_starts_dbg + (size_t)field * MAX_PULSES;
                    const int* field_pulse_lengths = h_pulse_lengths_dbg + (size_t)field * MAX_PULSES;
                    const int* field_pulse_types = h_pulse_types_dbg + (size_t)field * MAX_PULSES;
                    HostK3MatchDetail detail = replay_k3_match_for_line(
                        h_k3_debug[field], field_pulse_starts, field_pulse_types, ll_line, fmt.lines_per_frame);
                    if (detail.valid) {
                        fprintf(stderr,
                                "    [K3 host disagree replay] f%d ll=%d exp=%.1f match=%s pos=%d dist=%.1f tol=%.1f slot=%d cursor=%d->%d\n",
                                field, ll_line, detail.expected,
                                detail.matched ? "yes" : "no",
                                detail.chosen_pos, detail.match_dist, detail.max_allowed_distance,
                                detail.chosen_hsync_slot, detail.cursor_before, detail.cursor_after);
                        if (detail.chosen_hsync_slot >= 0) {
                            int hsync_starts[MAX_PULSES];
                            int hsync_lengths[MAX_PULSES];
                            int hsync_src_idx[MAX_PULSES];
                            int num_hsyncs = 0;
                            int npc_clamped = std::max(0, std::min(h_pulse_counts_dbg[field], MAX_PULSES));
                            for (int p = 0; p < npc_clamped && num_hsyncs < MAX_PULSES; p++) {
                                if (field_pulse_types[p] == PULSE_HSYNC) {
                                    hsync_starts[num_hsyncs] = field_pulse_starts[p];
                                    hsync_lengths[num_hsyncs] = field_pulse_lengths[p];
                                    hsync_src_idx[num_hsyncs] = p;
                                    num_hsyncs++;
                                }
                            }
                            int slot_start = std::max(0, detail.chosen_hsync_slot - 2);
                            int slot_end = std::min(num_hsyncs - 1, detail.chosen_hsync_slot + 3);
                            double chosen_len = (double)hsync_lengths[detail.chosen_hsync_slot];
                            fprintf(stderr,
                                    "    [K3 host disagree chosen] f%d ll=%d len=%.1f bucket=%s kind=%s\n",
                                    field, ll_line, chosen_len,
                                    width_bucket(chosen_len, fmt.hsync_width + h_k3_debug[field].hsync_offset),
                                    classify_host_pulse_kind(chosen_len, h_k3_debug[field], fmt));
                            fprintf(stderr, "    [K3 host disagree pulses] f%d ll=%d:", field, ll_line);
                            for (int slot = slot_start; slot <= slot_end; slot++) {
                                int src_idx = hsync_src_idx[slot];
                                double delta = (double)hsync_starts[slot] - detail.expected;
                                fprintf(stderr, " slot%d[src=%d %s/%s start=%d len=%d d=%.1f]%s",
                                        slot, src_idx, pulse_type_name(field_pulse_types[src_idx]),
                                        classify_host_pulse_kind((double)hsync_lengths[slot], h_k3_debug[field], fmt),
                                        hsync_starts[slot], hsync_lengths[slot], delta,
                                        (slot == detail.chosen_hsync_slot) ? "*" : "");
                            }
                            fprintf(stderr, "\n");
                            double prev_loc = (ll_line > 0) ? h_linelocs_coarse_dbg[(size_t)field * fmt.lines_per_frame + (size_t)ll_line - 1]
                                                            : detail.expected - h_k3_debug[field].meanlinelen;
                            double best_alt_score = 1.0e30;
                            int best_alt_slot = -1;
                            for (int slot = slot_start; slot <= slot_end; slot++) {
                                double pos = (double)hsync_starts[slot];
                                double dist = fabs(pos - detail.expected);
                                double spacing = pos - prev_loc;
                                double len_score = fabs((double)hsync_lengths[slot] - (fmt.hsync_width + h_k3_debug[field].hsync_offset));
                                double score = dist + 1.5 * fabs(spacing - h_k3_debug[field].meanlinelen) + 2.0 * len_score;
                                if (score < best_alt_score) {
                                    best_alt_score = score;
                                    best_alt_slot = slot;
                                }
                            }
                            if (best_alt_slot >= 0 && best_alt_slot != detail.chosen_hsync_slot) {
                                fprintf(stderr,
                                        "    [K3 host disagree alt] f%d ll=%d alt_slot=%d start=%d len=%d bucket=%s kind=%s\n",
                                        field, ll_line, best_alt_slot,
                                        hsync_starts[best_alt_slot], hsync_lengths[best_alt_slot],
                                        width_bucket((double)hsync_lengths[best_alt_slot],
                                                     fmt.hsync_width + h_k3_debug[field].hsync_offset),
                                        classify_host_pulse_kind((double)hsync_lengths[best_alt_slot], h_k3_debug[field], fmt));
                            }
                        }
                    }
                }
            }
            for (int i = 0; i < 4; i++) {
                if (top_hits[i].count <= 0) break;
                int field = top_hits[i].field;
                int out_line = top_hits[i].line;
                int ll_line = out_line + fmt.active_line_start;
                if (field < 0 || field >= fields_loaded || ll_line <= 0 || ll_line + 1 >= fmt.lines_per_frame) continue;
                const double* locs = h_linelocs_dbg + (size_t)field * fmt.lines_per_frame;
                const double* coarse_locs = h_linelocs_coarse_dbg + (size_t)field * fmt.lines_per_frame;
                double prev_spacing = locs[ll_line] - locs[ll_line - 1];
                double this_spacing = locs[ll_line + 1] - locs[ll_line];
                double next_spacing = (ll_line + 2 < fmt.lines_per_frame) ? (locs[ll_line + 2] - locs[ll_line + 1]) : 0.0;
                double coarse_prev_spacing = coarse_locs[ll_line] - coarse_locs[ll_line - 1];
                double coarse_this_spacing = coarse_locs[ll_line + 1] - coarse_locs[ll_line];
                double coarse_next_spacing = (ll_line + 2 < fmt.lines_per_frame) ? (coarse_locs[ll_line + 2] - coarse_locs[ll_line + 1]) : 0.0;
                fprintf(stderr,
                        "    [K5 top line detail] f%d out_line=%d ll=%d count=%d prev=%.1f this=%.1f next=%.1f nominal=%d\n",
                        field, out_line, ll_line, top_hits[i].count,
                        prev_spacing, this_spacing, next_spacing, fmt.samples_per_line);
                fprintf(stderr,
                        "    [K5 coarse/refined] f%d ll=%d coarse=(%.1f,%.1f,%.1f) refined=(%.1f,%.1f,%.1f) delta=(%.1f,%.1f,%.1f)\n",
                        field, ll_line,
                        coarse_prev_spacing, coarse_this_spacing, coarse_next_spacing,
                        prev_spacing, this_spacing, next_spacing,
                        prev_spacing - coarse_prev_spacing,
                        this_spacing - coarse_this_spacing,
                        next_spacing - coarse_next_spacing);
                if (debug_k3_host_ref && field >= 0 && field < (int)host_ref_tracks.size()) {
                    const HostRefTrackResult& ref = host_ref_tracks[(size_t)field];
                    if (ref.valid && ll_line > 0 && ll_line + 2 < fmt.lines_per_frame) {
                        const double* ref_locs = ref.locs.data();
                        double ref_prev_spacing = ref_locs[ll_line] - ref_locs[ll_line - 1];
                        double ref_this_spacing = ref_locs[ll_line + 1] - ref_locs[ll_line];
                        double ref_next_spacing = ref_locs[ll_line + 2] - ref_locs[ll_line + 1];
                        fprintf(stderr,
                                "    [K3 host ref] f%d ll=%d ref=(%.1f,%.1f,%.1f) vs_coarse=(%.1f,%.1f,%.1f)\n",
                                field, ll_line,
                                ref_prev_spacing, ref_this_spacing, ref_next_spacing,
                                ref_prev_spacing - coarse_prev_spacing,
                                ref_this_spacing - coarse_this_spacing,
                                ref_next_spacing - coarse_next_spacing);
                    }
                }
                if (h_k3_debug && h_pulse_counts_dbg && h_pulse_starts_dbg &&
                    h_pulse_lengths_dbg && h_pulse_types_dbg) {
                    const int* field_pulse_starts = h_pulse_starts_dbg + (size_t)field * MAX_PULSES;
                    const int* field_pulse_lengths = h_pulse_lengths_dbg + (size_t)field * MAX_PULSES;
                    const int* field_pulse_types = h_pulse_types_dbg + (size_t)field * MAX_PULSES;
                    HostK3MatchDetail detail = replay_k3_match_for_line(
                        h_k3_debug[field],
                        field_pulse_starts,
                        field_pulse_types,
                        ll_line,
                        fmt.lines_per_frame);
                    if (detail.valid) {
                        fprintf(stderr,
                                "    [K3 match replay] f%d ll=%d exp=%.1f match=%s pos=%d dist=%.1f tol=%.1f slot=%d cursor=%d->%d prev_h=%d next_h=%d npc=%d num_hsyncs=%d pulses=%d\n",
                                field, ll_line, detail.expected,
                                detail.matched ? "yes" : "no",
                                detail.chosen_pos, detail.match_dist, detail.max_allowed_distance,
                                detail.chosen_hsync_slot,
                                detail.cursor_before, detail.cursor_after,
                                detail.prev_hsync, detail.next_hsync,
                                h_k3_debug[field].npc, h_k3_debug[field].num_hsyncs,
                                h_pulse_counts_dbg[field]);
                        if (detail.chosen_hsync_slot >= 0) {
                            int hsync_starts[MAX_PULSES];
                            int hsync_lengths[MAX_PULSES];
                            int hsync_src_idx[MAX_PULSES];
                            int num_hsyncs = 0;
                            int npc_clamped = std::max(0, std::min(h_pulse_counts_dbg[field], MAX_PULSES));
                            for (int p = 0; p < npc_clamped && num_hsyncs < MAX_PULSES; p++) {
                                if (field_pulse_types[p] == PULSE_HSYNC) {
                                    hsync_starts[num_hsyncs] = field_pulse_starts[p];
                                    hsync_lengths[num_hsyncs] = field_pulse_lengths[p];
                                    hsync_src_idx[num_hsyncs] = p;
                                    num_hsyncs++;
                                }
                            }
                            int slot_start = std::max(0, detail.chosen_hsync_slot - 2);
                            int slot_end = std::min(num_hsyncs - 1, detail.chosen_hsync_slot + 2);
                            fprintf(stderr, "    [K3 pulse neighborhood] f%d ll=%d:", field, ll_line);
                            for (int slot = slot_start; slot <= slot_end; slot++) {
                                int src_idx = hsync_src_idx[slot];
                                double delta = (double)hsync_starts[slot] - detail.expected;
                                fprintf(stderr, " slot%d[src=%d %s start=%d len=%d d=%.1f]%s",
                                        slot, src_idx, pulse_type_name(field_pulse_types[src_idx]),
                                        hsync_starts[slot], hsync_lengths[slot], delta,
                                        (slot == detail.chosen_hsync_slot) ? "*" : "");
                            }
                            fprintf(stderr, "\n");

                            int src_center = hsync_src_idx[detail.chosen_hsync_slot];
                            int raw_start = std::max(0, src_center - 3);
                            int raw_end = std::min(npc_clamped - 1, src_center + 3);
                            fprintf(stderr, "    [K3 raw pulses] f%d ll=%d:", field, ll_line);
                            for (int p = raw_start; p <= raw_end; p++) {
                                double delta = (double)field_pulse_starts[p] - detail.expected;
                                fprintf(stderr, " p%d[%s start=%d len=%d d=%.1f]%s",
                                        p, pulse_type_name(field_pulse_types[p]),
                                        field_pulse_starts[p], field_pulse_lengths[p], delta,
                                        (p == src_center) ? "*" : "");
                            }
                            fprintf(stderr, "\n");
                        }
                    }
                }
            }
            delete[] h_pulse_counts_dbg;
            delete[] h_pulse_starts_dbg;
            delete[] h_pulse_lengths_dbg;
            delete[] h_pulse_types_dbg;
            delete[] h_linelocs_coarse_dbg;
            delete[] h_linelocs_dbg;
            delete[] h_line_counts;
        }
    }

    if (debug_k5_compare_coarse && !skip_k4) {
        cudaMemset(d_k5_coarse_bad_geom_line_count, 0, sizeof(int));
        cudaMemset(d_k5_coarse_sync_like_pixel_count, 0, sizeof(int));
        tbc_resample(static_cast<double*>(d_demod),
                     static_cast<double*>(d_linelocs_coarse),
                     static_cast<uint16_t*>(d_tbc_luma),
                     nullptr,
                     static_cast<int*>(d_k5_coarse_bad_geom_line_count),
                     static_cast<int*>(d_k5_coarse_sync_like_pixel_count),
                     nullptr,
                     false,
                     false,
                     fields_loaded,
                     fields_loaded * (int)spf_padded,
                     fmt);
        int h_k5_coarse_bad_geom = 0;
        int h_k5_coarse_sync_like = 0;
        cudaMemcpy(&h_k5_coarse_bad_geom, d_k5_coarse_bad_geom_line_count, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_k5_coarse_sync_like, d_k5_coarse_sync_like_pixel_count, sizeof(int), cudaMemcpyDeviceToHost);
        fprintf(stderr, "  [K5 coarse compare] chunk raw_offset=%zu: bad_geom_lines=%d sync_like_pixels=%d\n",
                raw_offset, h_k5_coarse_bad_geom, h_k5_coarse_sync_like);
    }

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
    delete[] h_k3_debug;
    if (d_k3_debug) cudaFree(d_k3_debug);
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
