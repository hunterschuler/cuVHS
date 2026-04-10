#!/usr/bin/env python3

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _setup_imports() -> None:
    repo = _repo_root()
    vhsdecode_root = repo / "vhs-decode"
    sys.path.insert(0, str(vhsdecode_root))
    candidates = [
        repo.parent / "VHSpp",
        repo.parent / "VHSpp" / "build",
        vhsdecode_root / "build",
        repo.parent / "vhs-decode" / "build",
    ]
    version_tag = f"cpython-{sys.version_info.major}{sys.version_info.minor}"
    found = []
    for base in candidates:
        if not base.exists():
            continue
        for path in sorted(base.rglob("vhsd_rust*.so")):
            found.append(path)
    for path in found:
        if version_tag in path.name:
            sys.path.append(str(path.parent))
            return
    if found:
        sys.path.append(str(found[0].parent))


def _write_u16(path: Path, arr) -> None:
    np.asarray(arr, dtype=np.uint16).tofile(path)


def _write_f64(path: Path, arr) -> None:
    np.asarray(arr, dtype=np.float64).tofile(path)


def _write_i32(path: Path, arr) -> None:
    np.asarray(arr, dtype=np.int32).tofile(path)


def _jsonable(value):
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return "1" if value else "0"
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value)
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return str(value)


def _build_decoder_options(args):
    import vhsdecode.formats as vf
    from vhsdecode.cmdcommons import get_extra_options, get_rf_options

    class Opts:
        pass

    opts = Opts()
    opts.chroma_trap = False
    opts.sharpness = 0
    opts.notch = None
    opts.notch_q = 10.0
    opts.debug = False
    opts.wow_level_adjust_smoothing = None
    opts.wow_interpolation_method = "linear"
    opts.dod_threshold_a = None
    opts.dod_hysteresis = 1.25
    opts.track_phase = None
    opts.high_boost = None
    opts.disable_diff_demod = False
    opts.fm_audio_notch = 0
    opts.enable_dc_offset = args.dc_correct
    opts.disable_comb = False
    opts.skip_chroma = False
    opts.nldeemp = False
    opts.subdeemp = False
    opts.y_comb = 0
    opts.cafc = False
    opts.disable_right_hsync = False
    opts.level_detect_divisor = 3
    opts.fallback_vsync = False
    opts.relaxed_line0 = False
    opts.field_order_confidence = 100
    opts.saved_levels = False
    opts.skip_hsync_refine = False
    opts.export_raw_tbc = False
    opts.tape_speed = args.tape_speed
    opts.ire0_adjust = False
    opts.detect_chroma_track_phase = False
    opts.disable_burst_hsync = False
    opts.disable_phase_correction = False
    opts.gnrc_afe = False
    opts.params_file = None
    opts.nodod = False
    opts.field_order_action = "detect"
    opts.AGC = False
    opts.noAGC = False

    rf_options = get_rf_options(opts)
    dod_threshold_p = vf.DEFAULT_THRESHOLD_P_CXADC if args.cxadc else vf.DEFAULT_THRESHOLD_P_DDD
    rf_options["dod_threshold_p"] = dod_threshold_p
    rf_options["dod_threshold_a"] = opts.dod_threshold_a
    rf_options["dod_hysteresis"] = opts.dod_hysteresis
    rf_options["track_phase"] = opts.track_phase
    rf_options["high_boost"] = opts.high_boost
    rf_options["disable_diff_demod"] = opts.disable_diff_demod
    rf_options["fm_audio_notch"] = opts.fm_audio_notch
    rf_options["disable_dc_offset"] = not opts.enable_dc_offset
    rf_options["disable_comb"] = opts.disable_comb
    rf_options["skip_chroma"] = opts.skip_chroma
    rf_options["nldeemp"] = opts.nldeemp
    rf_options["subdeemp"] = opts.subdeemp
    rf_options["y_comb"] = opts.y_comb
    rf_options["cafc"] = opts.cafc
    rf_options["disable_right_hsync"] = opts.disable_right_hsync
    rf_options["level_detect_divisor"] = opts.level_detect_divisor
    rf_options["fallback_vsync"] = opts.fallback_vsync
    rf_options["relaxed_line0"] = opts.relaxed_line0
    rf_options["field_order_confidence"] = int(max(0, min(100, opts.field_order_confidence)))
    rf_options["saved_levels"] = opts.saved_levels
    rf_options["skip_hsync_refine"] = opts.skip_hsync_refine
    rf_options["export_raw_tbc"] = opts.export_raw_tbc
    rf_options["tape_speed"] = opts.tape_speed
    rf_options["ire0_adjust"] = opts.ire0_adjust
    rf_options["detect_chroma_track_phase"] = opts.detect_chroma_track_phase
    rf_options["disable_burst_hsync"] = opts.disable_burst_hsync
    rf_options["disable_phase_correction"] = opts.disable_phase_correction
    rf_options["gnrc_afe"] = opts.gnrc_afe

    extra_options = get_extra_options(opts, True)
    extra_options["params_file"] = opts.params_file
    return rf_options, extra_options, (not opts.nodod), opts.field_order_action


def main() -> int:
    _setup_imports()

    import lddecode.utils as lddu
    import vhsdecode.addons.vsyncserration as vsyncserration
    from vhsdecode.process import VHSDecode
    from vhsdecode.chroma import (
        get_burst_area,
        burst_deemphasis,
        upconvert_chroma,
        ntsc_phase_comp,
        comb_c_ntsc,
        comb_c_pal,
    )
    from vhsdecode.rust_utils import sosfiltfilt_rust
    import lddecode.core as ldd

    vsyncserration.chainfiltfilt_b = lambda data, filters: vsyncserration._chainfiltfilt(data, filters)

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample", type=int, required=True)
    parser.add_argument("--system", default="NTSC")
    parser.add_argument("--tape-format", default="VHS")
    parser.add_argument("--tape-speed", default="sp")
    parser.add_argument("--inputfreq", type=float, default=28.0)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--no-resample", action="store_true", default=True)
    parser.add_argument("--cxadc", action="store_true", default=True)
    parser.add_argument("--dc-correct", action="store_true", default=False)
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("dump_k4_vhsdecode")
    logger.handlers.clear()
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)

    loader_input_freq = args.inputfreq if not args.no_resample else None
    sample_freq = 40.0 if not args.no_resample else args.inputfreq
    loader = lddu.make_loader(args.input, loader_input_freq)
    rf_options, extra_options, do_dod, field_order_action = _build_decoder_options(args)
    decoder = VHSDecode(
        args.input,
        None,
        loader,
        logger,
        system=args.system,
        tape_format=args.tape_format,
        doDOD=do_dod,
        threads=args.threads,
        inputfreq=sample_freq,
        rf_options=rf_options,
        extra_options=extra_options,
        field_order_action=field_order_action,
    )

    f, offset = decoder.decodefield(args.sample, decoder.mtf_level, None, False, False)
    if f is None:
        raise RuntimeError("decodefield returned no field")
    if not getattr(f, "valid", False):
        raise RuntimeError("decodefield returned invalid field")

    (luma, chroma), _, _ = f.downscale(final=True)
    _write_u16(outdir / "python_luma_u16.u16", luma)
    if chroma is not None:
        _write_u16(outdir / "python_chroma_u16.u16", chroma)
    _write_f64(outdir / "python_linelocs_final.f64", f.linelocs)
    _write_f64(outdir / "python_demod_window.f64", f.data["video"]["demod"])
    _write_f64(outdir / "python_demod_burst_window.f64", f.data["video"]["demod_burst"])

    # Detailed chroma stage dumps for K4 accountability.
    chroma_tbc, _, _ = ldd.Field.downscale(f, channel="demod_burst")
    _write_f64(outdir / "python_chroma_tbc.f64", chroma_tbc)
    source_lineoffset = f.lineoffset
    lineoffset = source_lineoffset + 1
    linesout = f.outlinecount
    outwidth = f.outlinelen
    outsamples = linesout * outwidth
    outline_offset = (source_lineoffset + 1) * outwidth
    if hasattr(f, "interpolated_pixel_locs") and hasattr(f, "wowfactors"):
        interp = np.asarray(f.interpolated_pixel_locs, dtype=np.float64)
        wow = np.asarray(f.wowfactors, dtype=np.float64)
        if interp.size >= outline_offset + outsamples and wow.size >= outline_offset + outsamples:
            wow_median = np.median(wow)
            wow_mad = np.median(np.abs(wow - wow_median))
            wow_threshold = 15.0 * wow_mad if wow_mad > 0.0 else 0.001
            wow_level_adjust = np.where(np.abs(wow - wow_median) > wow_threshold, wow_median, wow)
            _write_f64(
                outdir / "python_chroma_source_coords.f64",
                interp[outline_offset : outline_offset + outsamples],
            )
            _write_f64(
                outdir / "python_chroma_source_level_adjust.f64",
                wow_level_adjust[outline_offset : outline_offset + outsamples],
            )
    burst_area_init = get_burst_area(f)
    burstarea = burst_area_init[0] - 5, burst_area_init[1] + 10

    chroma_work = chroma_tbc.copy()
    if f.rf.color_system == "NTSC":
        chroma_work = burst_deemphasis(chroma_work, lineoffset, linesout, outwidth, burstarea)
    _write_f64(outdir / "python_chroma_postdeemph.f64", chroma_work)

    chroma_heterodyne = (
        f.rf.chroma_afc.getChromaHet()
        if (f.rf.do_cafc and not False)
        else f.rf.chroma_heterodyne
    )
    uphet_prephase = upconvert_chroma(
        chroma_work,
        lineoffset,
        outwidth,
        chroma_heterodyne,
        f.phase_sequence,
    )
    _write_f64(outdir / "python_chroma_uphet_prephase.f64", uphet_prephase)
    phase_per_line = np.full(linesout, -1, dtype=np.int32)
    for row in f.phase_sequence:
        if len(row) < 2:
            continue
        line_number = int(row[0])
        current_phase = int(row[1])
        line_idx = line_number - lineoffset
        if 0 <= line_idx < linesout:
            phase_per_line[line_idx] = current_phase
    _write_i32(outdir / "python_phase_index_per_line.i32", phase_per_line)

    het_len = linesout * outwidth
    het_selected = np.zeros(het_len, dtype=np.float64)
    for phase in range(4):
        het_phase = np.asarray(chroma_heterodyne[phase][:het_len], dtype=np.float64)
        _write_f64(outdir / f"python_chroma_heterodyne_p{phase}.f64", het_phase)
    for line_idx in range(linesout):
        phase = int(phase_per_line[line_idx])
        if phase < 0 or phase > 3:
            continue
        start = line_idx * outwidth
        end = start + outwidth
        het_selected[start:end] = chroma_heterodyne[phase][start:end]
    _write_f64(outdir / "python_chroma_heterodyne_selected.f64", het_selected)

    uphet_postphase = uphet_prephase.copy()
    if f.rf.color_system == "NTSC" and not f.rf.options.disable_phase_correction:
        uphet_postphase = ntsc_phase_comp(uphet_postphase, f.burst_phase_avg)
    _write_f64(outdir / "python_chroma_postphase.f64", uphet_postphase)

    uphet_postfilter = sosfiltfilt_rust(f.rf.Filters["FChromaFinal"], uphet_postphase.copy())
    _write_f64(outdir / "python_chroma_postfilter.f64", uphet_postfilter)

    uphet_preacc = uphet_postfilter.copy()
    if not f.rf.options.disable_comb:
        if f.rf.color_system == "NTSC":
            uphet_preacc = comb_c_ntsc(uphet_preacc, outwidth)
        else:
            uphet_preacc = comb_c_pal(uphet_preacc, outwidth)
    _write_f64(outdir / "python_chroma_preacc.f64", uphet_preacc)

    meta = {
        "requested_start": int(args.sample),
        "decodefield_offset": _jsonable(offset),
        "field_readloc": int(getattr(f, "readloc", 0)),
        "blocklen": _jsonable(getattr(f.rf, "blocklen", None)),
        "blockcut": _jsonable(getattr(f.rf, "blockcut", None)),
        "blockcut_end": _jsonable(getattr(f.rf, "blockcut_end", None)),
        "luma_count": int(len(luma)),
        "chroma_count": int(len(chroma)) if chroma is not None else 0,
        "meanlinelen": _jsonable(getattr(f, "meanlinelen", None)),
        "first_hsync_loc": _jsonable(getattr(f, "first_hsync_loc", None)),
        "first_hsync_loc_line": _jsonable(getattr(f, "first_hsync_loc_line", None)),
        "isFirstField": _jsonable(getattr(f, "isFirstField", None)),
        "field_number": _jsonable(getattr(f, "field_number", None)),
        "track_phase": _jsonable(getattr(f.rf, "track_phase", None)),
        "phase_sequence": _jsonable(getattr(f, "phase_sequence", None)),
        "line_count": int(len(f.linelocs)),
    }
    with (outdir / "config.json").open("w", encoding="utf-8") as fcfg:
        json.dump(meta, fcfg, indent=2, sort_keys=True)
    print(json.dumps(meta, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
