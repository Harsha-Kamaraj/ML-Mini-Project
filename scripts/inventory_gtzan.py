#!/usr/bin/env python3
"""Inspect original GTZAN audio without modifying, segmenting, or excluding it.

Requires NumPy and SoundFile. Discovery uses the ten canonical genre names in
directory components, with a filename-prefix fallback. Other directories are
not treated as new genres. Recording IDs are ``genre.basename_without_suffix``
with an existing matching genre prefix removed to avoid duplication, e.g.
``genres/jazz/jazz.00000.au`` has ID ``jazz.00000``.

Files named ``._*`` are classified as AppleDouble metadata only when their
first four bytes equal magic 0x00051607. They remain untouched and are listed
separately in JSON, not counted as recordings. Unverified ``._*`` files are
reported as hard failures; audio-suffix candidates still undergo decoding.

All audio is decoded in blocks, not merely probed. Amplitude statistics describe
SoundFile's float32 decoder output (integer PCM is normalized by libsndfile).
Warnings are inspection heuristics, never automatic exclusion rules:
duration outside [29, 31] seconds; unexpected rate/channels; peak > 1.000001;
RMS < 1e-6; >99% exact zeros; >=1% samples with |amplitude| >= .9999.
The last condition is a near-full-scale warning, not proof of clipping.

Exit 0: expected selected IDs are present and no hard failures were found.
Exit 2: missing selected IDs, duplicate IDs, unknown/mismatched genre labels,
decode/hash failure, nonfinite samples, or header/decoded frame disagreement.
Warnings may remain even on exit 0; read the JSON summary. No file is excluded.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf


GENRES = ("blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock")
SELECTED_GENRES = ("classical", "jazz", "metal", "pop")
AUDIO_SUFFIXES = {".au", ".wav", ".aif", ".aiff", ".flac", ".ogg", ".mp3"}
APPLEDOUBLE_MAGIC = bytes.fromhex("00051607")
PREVIEW_SECONDS = 0.05
CSV_FIELDS = (
    "relative_path", "recording_id", "genre", "selected_genre", "sample_rate",
    "channels", "duration", "decode_status", "format", "subtype", "header_frames",
    "decoded_frames", "file_bytes", "sha256", "sample_count", "finite_sample_count",
    "nonfinite_sample_count", "minimum", "maximum", "peak", "rms", "zero_fraction",
    "near_full_scale_fraction", "anomalies", "error",
)
HARD_ANOMALIES = {
    "unknown_genre", "ambiguous_genre_directory", "genre_filename_mismatch",
    "duplicate_recording_id", "hash_failed", "nonfinite_samples", "frame_count_mismatch",
}


def identify(path: Path, root: Path) -> tuple[str, str, list[str]]:
    """Prefer explicit genre directory labels and flag any naming conflict."""
    relative = path.relative_to(root)
    directory_genres = set(relative.parts[:-1]) & set(GENRES)
    prefix = path.stem.split(".", 1)[0].lower()
    anomalies = []
    if len(directory_genres) == 1:
        genre = next(iter(directory_genres))
        if prefix in GENRES and prefix != genre:
            anomalies.append("genre_filename_mismatch")
    elif len(directory_genres) > 1:
        genre = "unknown"
        anomalies.append("ambiguous_genre_directory")
    else:
        genre = prefix if prefix in GENRES else "unknown"
        if genre == "unknown":
            anomalies.append("unknown_genre")
    stem = path.stem
    recording_id = stem if stem.startswith(genre + ".") else f"{genre}.{stem}"
    return genre, recording_id, anomalies


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_audio(path: Path, root: Path) -> tuple[dict, np.ndarray | None]:
    """Return a row even if opening or decoding fails; never change raw audio."""
    genre, recording_id, anomalies = identify(path, root)
    row = dict.fromkeys(CSV_FIELDS, None)
    row.update(relative_path=path.relative_to(root).as_posix(), recording_id=recording_id,
               genre=genre, selected_genre=genre in SELECTED_GENRES,
               decode_status="not_attempted", anomalies=anomalies, error="")
    errors = []
    try:
        row["file_bytes"] = path.stat().st_size
        row["sha256"] = sha256_file(path)
    except OSError as exc:
        anomalies.append("hash_failed")
        errors.append(f"{type(exc).__name__}: {exc}")

    preview = None
    count = finite_count = zero_count = near_full_scale_count = frames = 0
    sum_squares = 0.0
    minimum, maximum = math.inf, -math.inf
    try:
        with sf.SoundFile(path) as audio:
            row.update(sample_rate=audio.samplerate, channels=audio.channels,
                       header_frames=audio.frames, duration=audio.frames / audio.samplerate,
                       format=audio.format, subtype=audio.subtype)
            preview_size = max(1, round(audio.samplerate * PREVIEW_SECONDS))
            for block in audio.blocks(blocksize=65536, dtype="float32", always_2d=True):
                frames += len(block)
                if preview is None:
                    preview = block[:preview_size].copy()
                values = block.reshape(-1)
                count += values.size
                finite = values[np.isfinite(values)]
                finite_count += finite.size
                if finite.size:
                    minimum = min(minimum, float(finite.min()))
                    maximum = max(maximum, float(finite.max()))
                    sum_squares += float(np.square(finite.astype(np.float64)).sum())
                    zero_count += int(np.count_nonzero(finite == 0))
                    near_full_scale_count += int(np.count_nonzero(np.abs(finite) >= 0.9999))
            row["decode_status"] = "ok"
            if frames != audio.frames:
                anomalies.append("frame_count_mismatch")
            if frames == 0:
                row["decode_status"] = "empty"
                anomalies.append("empty_audio")
    except (RuntimeError, OSError, ValueError) as exc:
        row["decode_status"] = "failed"
        errors.append(f"{type(exc).__name__}: {exc}")

    row.update(decoded_frames=frames, sample_count=count, finite_sample_count=finite_count,
               nonfinite_sample_count=count - finite_count)
    if finite_count:
        row.update(minimum=minimum, maximum=maximum, peak=max(abs(minimum), abs(maximum)),
                   rms=math.sqrt(sum_squares / finite_count), zero_fraction=zero_count / finite_count,
                   near_full_scale_fraction=near_full_scale_count / finite_count)
    if count != finite_count:
        anomalies.append("nonfinite_samples")
    if row["sample_rate"] is not None and row["sample_rate"] != 22050:
        anomalies.append("unexpected_sample_rate")
    if row["channels"] is not None and row["channels"] != 1:
        anomalies.append("unexpected_channels")
    if row["duration"] is not None and not 29 <= row["duration"] <= 31:
        anomalies.append("unusual_duration")
    if row["peak"] is not None and row["peak"] > 1.000001:
        anomalies.append("outside_normalized_amplitude_range")
    if row["rms"] is not None and row["rms"] < 1e-6:
        anomalies.append("near_silent_audio")
    if row["zero_fraction"] is not None and row["zero_fraction"] > 0.99:
        anomalies.append("mostly_zero_audio")
    if row["near_full_scale_fraction"] is not None and row["near_full_scale_fraction"] >= 0.01:
        anomalies.append("frequent_near_full_scale_samples")
    row["error"] = "; ".join(errors)
    return row, preview


def describe(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "min": None, "median": None, "max": None}
    return {"count": len(values), "min": min(values), "median": float(np.median(values)), "max": max(values)}


def summarize(rows: list[dict], root: Path, ignored: list[str]) -> dict:
    by_id, by_hash = defaultdict(list), defaultdict(list)
    for row in rows:
        by_id[row["recording_id"]].append(row)
        if row["sha256"]:
            by_hash[row["sha256"]].append(row)
    duplicate_ids = {key: [r["relative_path"] for r in group] for key, group in by_id.items() if len(group) > 1}
    duplicate_hashes = {key: [r["relative_path"] for r in group] for key, group in by_hash.items() if len(group) > 1}
    for key in duplicate_ids:
        for row in by_id[key]:
            row["anomalies"].append("duplicate_recording_id")
    for key in duplicate_hashes:
        for row in by_hash[key]:
            row["anomalies"].append("duplicate_file_bytes")

    genre_summary = {}
    for genre in GENRES:
        group = [r for r in rows if r["genre"] == genre]
        observed_ids = {r["recording_id"] for r in group}
        expected_ids = {f"{genre}.{index:05d}" for index in range(100)}
        genre_summary[genre] = {
            "files_found": len(group), "decoded_ok": sum(r["decode_status"] == "ok" for r in group),
            "decode_failures": sum(r["decode_status"] != "ok" for r in group),
            "expected_count": 100, "missing_expected_ids": sorted(expected_ids - observed_ids),
            "unexpected_ids": sorted(observed_ids - expected_ids),
            "sample_rates": dict(Counter(str(r["sample_rate"]) for r in group)),
            "channel_counts": dict(Counter(str(r["channels"]) for r in group)),
            "duration_seconds": describe([r["duration"] for r in group if r["duration"] is not None]),
        }
    hard_failures = []
    if not root.is_dir():
        hard_failures.append(f"Dataset root is not a directory: {root}")
    for genre in SELECTED_GENRES:
        missing = genre_summary[genre]["missing_expected_ids"]
        if missing:
            hard_failures.append(f"{genre}: {len(missing)} missing expected recording IDs")
    for row in rows:
        if row["decode_status"] != "ok":
            hard_failures.append(f"{row['relative_path']}: decode_status={row['decode_status']}")
        for anomaly in sorted(set(row["anomalies"]) & HARD_ANOMALIES):
            hard_failures.append(f"{row['relative_path']}: {anomaly}")
    anomaly_counts = Counter(a for r in rows for a in r["anomalies"])
    return {
        "schema_version": 1, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root), "status": "failed" if hard_failures else ("passed_with_warnings" if anomaly_counts else "passed"),
        "scope": "Full native audio decode; no exclusions, segmentation, pooling, or splits.",
        "recording_id_rule": "genre + '.' + filename stem after removing an existing matching genre prefix",
        "selected_genres": list(SELECTED_GENRES), "expected_selected_recordings": 400,
        "total_audio_files": len(rows), "selected_audio_files": sum(r["selected_genre"] for r in rows),
        "decode_status_counts": dict(Counter(r["decode_status"] for r in rows)),
        "sample_rate_counts": dict(Counter(str(r["sample_rate"]) for r in rows)),
        "channel_counts": dict(Counter(str(r["channels"]) for r in rows)),
        "duration_seconds": describe([r["duration"] for r in rows if r["duration"] is not None]),
        "by_genre": genre_summary, "hard_failures": hard_failures,
        "anomaly_counts": dict(sorted(anomaly_counts.items())),
        "anomaly_files": [{"relative_path": r["relative_path"], "anomalies": r["anomalies"], "error": r["error"]}
                          for r in rows if r["anomalies"] or r["decode_status"] != "ok"],
        "duplicate_recording_ids": duplicate_ids, "duplicate_file_sha256": duplicate_hashes,
        "ignored_non_audio_files": ignored,
        "thresholds": {"expected_sample_rate": 22050, "expected_channels": 1,
                       "ordinary_duration_seconds_inclusive": [29, 31], "out_of_range_absolute_peak_gt": 1.000001,
                       "near_silence_rms_lt": 1e-6, "mostly_zero_fraction_gt": 0.99,
                       "near_full_scale_absolute_amplitude_gte": 0.9999,
                       "frequent_near_full_scale_fraction_gte": 0.01},
        "statistics_note": "All channels; normalized float32 decoder output. Amplitude statistics use finite decoded samples only; nonfinite counts are explicit. Duration is from header frames. Partial decoding is retained and marked failed.",
        "limitations": "Byte hashes detect exact duplicate files only. This audit cannot certify genre labels, distinct performances, absence of corruption that still decodes, or absence of data leakage.",
        "versions": {"numpy": np.__version__, "soundfile": sf.__version__, "libsndfile": sf.__libsndfile_version__},
    }


def write_waveforms(path: Path, previews: dict[str, tuple[dict, np.ndarray]]) -> None:
    """Write a directly viewable raw-amplitude SVG; no plotting package needed."""
    width, height = 1200, 690
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<g font-family="sans-serif" fill="#18242e">',
             '<text x="30" y="32" font-size="22">GTZAN: native raw waveform inspection</text>',
             '<text x="30" y="55" font-size="13">First 0.05 seconds; channel 1; fixed amplitude axis [-1, 1]. Display only, no preprocessing.</text>']
    colors = ("#2463a7", "#277e5c", "#94439c", "#b36314")
    for index, genre in enumerate(SELECTED_GENRES):
        ox, oy = 30 + (index % 2) * 590, 90 + (index // 2) * 300
        parts.append(f'<text x="{ox}" y="{oy}" font-size="19">{genre}</text>')
        if genre not in previews:
            parts.append(f'<text x="{ox}" y="{oy + 35}" font-size="14">No successfully decoded representative available</text>')
            continue
        row, data = previews[genre]
        rate = row["sample_rate"]
        x0, y0, plot_w, plot_h = ox + 43, oy + 55, 500, 140
        metadata = html.escape(f"{row['recording_id']} | {rate} Hz | {row['channels']} ch | {row['duration']:.3f} s")
        parts.append(f'<text x="{ox}" y="{oy + 23}" font-size="12">{metadata}</text>')
        parts.append(f'<rect x="{x0}" y="{y0}" width="{plot_w}" height="{plot_h}" fill="#f4f7fa" stroke="#99a4ae"/>')
        for amplitude in (-1, 0, 1):
            y = y0 + plot_h * (1 - amplitude) / 2
            parts.append(f'<line x1="{x0}" y1="{y}" x2="{x0 + plot_w}" y2="{y}" stroke="#cbd4dc"/>')
            parts.append(f'<text x="{x0 - 10}" y="{y + 4}" font-size="11" text-anchor="end">{amplitude}</text>')
        for time in (0, .025, .05):
            x = x0 + plot_w * time / PREVIEW_SECONDS
            parts.append(f'<text x="{x}" y="{y0 + plot_h + 20}" font-size="11" text-anchor="middle">{time:g}</text>')
        points = " ".join(f"{x0 + plot_w * frame / rate / PREVIEW_SECONDS:.3f},{y0 + plot_h * (1 - float(value)) / 2:.3f}"
                          for frame, value in enumerate(data[:, 0]) if math.isfinite(float(value)))
        parts.append(f'<clipPath id="panel{index}"><rect x="{x0}" y="{y0}" width="{plot_w}" height="{plot_h}"/></clipPath>')
        parts.append(f'<polyline points="{points}" fill="none" stroke="{colors[index]}" stroke-width=".8" clip-path="url(#panel{index})"/>')
        parts.append(f'<text x="{x0 + plot_w / 2}" y="{y0 + plot_h + 40}" font-size="12" text-anchor="middle">Time from recording start (seconds)</text>')
        detail = html.escape(f"Frames [0, {len(data)}); final displayed sample at {(len(data) - 1) / rate:.6f} s; full-file peak {row['peak']:.5f}")
        parts.append(f'<text x="{ox}" y="{oy + 263}" font-size="11">{detail}</text>')
    parts.append('</g></svg>')
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def run_inventory(root: Path, output_dir: Path) -> dict:
    root, output_dir = root.expanduser().resolve(), output_dir.expanduser().resolve()
    if output_dir == root or root in output_dir.parents:
        raise ValueError("Output directory must be outside the raw-audio root to preserve raw files.")
    output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(path for path in root.rglob("*") if path.is_file()) if root.is_dir() else []
    audio_paths, ignored, sidecars, unverified_sidecars = [], [], [], []
    for path in files:
        relative_path = path.relative_to(root).as_posix()
        if path.name.startswith("._"):
            try:
                with path.open("rb") as source:
                    magic = source.read(4)
                if magic == APPLEDOUBLE_MAGIC:
                    sidecars.append({"relative_path": relative_path, "header_magic_hex": magic.hex(),
                                     "file_bytes": path.stat().st_size})
                    continue
                unverified_sidecars.append({"relative_path": relative_path, "header_magic_hex": magic.hex(),
                                            "error": "._ filename without AppleDouble magic"})
            except OSError as exc:
                unverified_sidecars.append({"relative_path": relative_path, "header_magic_hex": None,
                                            "error": f"{type(exc).__name__}: {exc}"})
        if path.suffix.lower() in AUDIO_SUFFIXES:
            audio_paths.append(path)
        else:
            ignored.append(relative_path)
    rows, previews = [], {}
    for index, path in enumerate(audio_paths, start=1):
        row, preview = inspect_audio(path, root)
        rows.append(row)
        if row["genre"] in SELECTED_GENRES and row["genre"] not in previews and row["decode_status"] == "ok" and preview is not None:
            previews[row["genre"]] = (row, preview)
        if index % 100 == 0:
            print(f"Decoded {index}/{len(audio_paths)} audio files", flush=True)
    summary = summarize(rows, root, ignored)
    summary.update(
        total_discovered_files=len(files),
        appledouble_metadata_count=len(sidecars),
        appledouble_metadata_files=sidecars,
        appledouble_classification_rule="Filename starts with ._ AND first four bytes are 0x00051607. Preserved as metadata; not a recording or an audio exclusion.",
        unverified_appledouble_candidates=unverified_sidecars,
    )
    if unverified_sidecars:
        summary["status"] = "failed"
        summary["hard_failures"].extend(f"{item['relative_path']}: unverified AppleDouble candidate: {item['error']}"
                                        for item in unverified_sidecars)
    summary["waveform_representatives"] = {
        genre: {"relative_path": row["relative_path"], "recording_id": row["recording_id"],
                "start_frame": 0, "end_frame_exclusive": len(data), "channel_shown": 1,
                "sample_rate": row["sample_rate"]}
        for genre, (row, data) in previews.items()
    }
    with (output_dir / "gtzan_inventory.csv").open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "anomalies": ";".join(row["anomalies"])})
    (output_dir / "gtzan_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_waveforms(output_dir / "gtzan_waveforms.svg", previews)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("data/raw/gtzan"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/metadata"))
    args = parser.parse_args()
    try:
        summary = run_inventory(args.root, args.output_dir)
    except ValueError as exc:
        parser.error(str(exc))
    print(f"Status: {summary['status']}; audio files: {summary['total_audio_files']}; selected: {summary['selected_audio_files']}")
    print(f"Preserved AppleDouble metadata files: {summary['appledouble_metadata_count']}; other non-audio files: {len(summary['ignored_non_audio_files'])}")
    for genre in SELECTED_GENRES:
        result = summary["by_genre"][genre]
        print(f"  {genre}: {result['files_found']} found, {result['decoded_ok']} decoded, {len(result['missing_expected_ids'])} missing expected IDs")
    print(f"Anomalies: {summary['anomaly_counts']}; hard failures: {len(summary['hard_failures'])}")
    print(f"Wrote GTZAN inventory, summary and waveform SVG to {args.output_dir}")
    return 2 if summary["hard_failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
