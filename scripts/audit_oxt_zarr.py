#!/usr/bin/env python3
"""Read pressure-only Zarr arrays and episode metadata from one OXT object ZIP."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import zarr


def _strings(array: Any) -> np.ndarray:
    values = np.asarray(array[:])
    return np.char.strip(values.astype(str))


def _constant_or_values(values: np.ndarray) -> list[str]:
    return sorted(set(values.tolist()))


def audit_oxt_zarr_archive(archive_path: Path, object_name: str) -> dict:
    store = zarr.ZipStore(str(archive_path), mode="r")
    try:
        root = zarr.open_group(store=store, path=f"{object_name}.zarr", mode="r")
        data = root["data"]
        meta = root["meta"]

        episode_ends = np.asarray(meta["episode_ends"][:], dtype=np.int64)
        frame_count = int(data["timestamps"].shape[0])
        if episode_ends.ndim != 1 or len(episode_ends) == 0:
            raise ValueError("meta/episode_ends must be a non-empty 1D array")
        if np.any(np.diff(episode_ends) <= 0) or int(episode_ends[-1]) != frame_count:
            raise ValueError("episode_ends are not strictly increasing or do not end at frame_count")

        required = [
            "acquisition_mode", "pairing_quality", "subject_id",
            "timestamps", "tactile_timestamps",
            "right_tactile_data_pressure", "right_tactile_raw_pressure_256",
            "right_tactile_raw_index_pressure",
        ]
        missing = [name for name in required if name not in data]
        if missing:
            raise KeyError(f"Missing required data arrays: {missing}")

        acquisition = _strings(data["acquisition_mode"])
        pairing = _strings(data["pairing_quality"])
        subject = _strings(data["subject_id"])
        timestamps = np.asarray(data["tactile_timestamps"][:], dtype=np.float64)
        frame_timestamps = np.asarray(data["timestamps"][:], dtype=np.float64)
        pressure_162 = np.asarray(data["right_tactile_data_pressure"][:], dtype=np.float32)
        raw_256 = np.asarray(data["right_tactile_raw_pressure_256"][:])
        raw_indices = np.asarray(data["right_tactile_raw_index_pressure"][:], dtype=np.int64)

        if pressure_162.shape != (frame_count, 162):
            raise ValueError(f"Unexpected 162-layout pressure shape: {pressure_162.shape}")
        if raw_256.shape != (frame_count, 256):
            raise ValueError(f"Unexpected raw pressure shape: {raw_256.shape}")
        if raw_indices.shape != (frame_count, 162):
            raise ValueError(f"Unexpected raw index shape: {raw_indices.shape}")
        if any(len(x) != frame_count for x in (acquisition, pairing, subject, timestamps, frame_timestamps)):
            raise ValueError("Metadata/timestamp arrays are not frame-aligned")

        mapping_first = raw_indices[0]
        mapping_constant = bool(np.all(raw_indices == mapping_first[None, :]))
        unique_indices = sorted(set(mapping_first.tolist()))
        finite = np.isfinite(pressure_162)
        all_tactile_dt = np.diff(timestamps)
        duplicate_mask = all_tactile_dt == 0
        if duplicate_mask.any():
            duplicate_pressure_equal = np.all(
                pressure_162[1:][duplicate_mask] == pressure_162[:-1][duplicate_mask], axis=1
            )
            duplicate_pressure_equal_count = int(duplicate_pressure_equal.sum())
        else:
            duplicate_pressure_equal_count = 0
        pressure_stats = {
            "finite_fraction": float(finite.mean()),
            "min": float(np.nanmin(pressure_162)) if finite.any() else None,
            "max": float(np.nanmax(pressure_162)) if finite.any() else None,
            "p50": float(np.nanpercentile(pressure_162, 50)) if finite.any() else None,
            "p99": float(np.nanpercentile(pressure_162, 99)) if finite.any() else None,
            "negative_value_count": int(np.sum(pressure_162 < 0)),
        }

        episodes = []
        start = 0
        for episode_index, end_value in enumerate(episode_ends.tolist()):
            end = int(end_value)
            acq_values = _constant_or_values(acquisition[start:end])
            pair_values = _constant_or_values(pairing[start:end])
            subject_values = _constant_or_values(subject[start:end])
            ts = timestamps[start:end]
            dt = np.diff(ts)
            positive_dt = int(np.sum(dt > 0))
            zero_dt = int(np.sum(dt == 0))
            negative_dt = int(np.sum(dt < 0))
            monotonic = bool(len(ts) >= 2 and negative_dt == 0)
            median_dt = float(np.median(dt[dt > 0])) if positive_dt else None
            frame_ts = frame_timestamps[start:end]
            frame_dt = np.diff(frame_ts)
            duplicate_mask_ep = dt == 0
            if duplicate_mask_ep.any():
                duplicate_equal_ep = np.all(
                    pressure_162[start + 1:end][duplicate_mask_ep] ==
                    pressure_162[start:end - 1][duplicate_mask_ep], axis=1
                )
                duplicate_mismatch_count_ep = int((~duplicate_equal_ep).sum())
            else:
                duplicate_mismatch_count_ep = 0
            frame_time_ok = bool(len(frame_ts) >= 2 and np.all(frame_dt >= 0))
            frame_duration = float(frame_ts[-1] - frame_ts[0]) if len(frame_ts) >= 2 else 0.0
            unique_tactile_times = int(len(np.unique(ts[np.isfinite(ts)])))
            pressure_change_count = int(np.sum(np.any(np.diff(pressure_slice, axis=0) != 0, axis=1))) if len(pressure_slice) > 1 else 0
            timestamp_integrity_ok = bool(negative_dt == 0 and duplicate_mismatch_count_ep == 0 and frame_time_ok)
            duration = float(ts[-1] - ts[0]) if len(ts) >= 2 else 0.0
            is_gloved = acq_values == ["gloved_hand"]
            is_direct = pair_values == ["direct"]
            pressure_slice = pressure_162[start:end]
            pressure_ok = bool(np.isfinite(pressure_slice).all())
            episodes.append({
                "episode_index": episode_index,
                "start_frame": start,
                "end_frame_exclusive": end,
                "frame_count": end - start,
                "duration_seconds": duration,
                "frame_timestamp_duration_seconds": frame_duration,
                "unique_tactile_timestamp_count": unique_tactile_times,
                "consecutive_pressure_change_count": pressure_change_count,
                "sampling_interval_median_seconds": median_dt,
                "timestamps_non_decreasing": monotonic,
                "timestamp_positive_interval_count": positive_dt,
                "timestamp_duplicate_interval_count": zero_dt,
                "timestamp_negative_interval_count": negative_dt,
                "frame_timestamp_negative_interval_count": int(np.sum(frame_dt < 0)),
                "frame_timestamp_duplicate_interval_count": int(np.sum(frame_dt == 0)),
                "duplicate_timestamp_pressure_mismatch_count": duplicate_mismatch_count_ep,
                "timestamp_integrity_ok": timestamp_integrity_ok,
                "acquisition_mode_values": acq_values,
                "pairing_quality_values": pair_values,
                "subject_id_values": subject_values,
                "single_subject": len(subject_values) == 1,
                "gloved_hand_only": is_gloved,
                "direct_pairing_only": is_direct,
                "pressure_finite": pressure_ok,
                "preliminary_3s_window_eligible": bool(
                    is_gloved and is_direct and len(subject_values) == 1 and timestamp_integrity_ok and duration >= 3.0 and pressure_ok
                ),
                "eligibility_note": (
                    "Preliminary duration/signal check only; contact onset and target validity "
                    "have not yet been evaluated."
                ),
            })
            start = end

        eligible = [x for x in episodes if x["preliminary_3s_window_eligible"]]
        participant_counts: dict[str, int] = {}
        for ep in eligible:
            for sid in ep["subject_id_values"]:
                participant_counts[sid] = participant_counts.get(sid, 0) + 1

        return {
            "object_name": object_name,
            "archive": archive_path.name,
            "frame_count": frame_count,
            "episode_count": len(episodes),
            "episode_ends_last_value": int(episode_ends[-1]),
            "pressure_162_shape": list(pressure_162.shape),
            "raw_pressure_256_shape": list(raw_256.shape),
            "raw_index_mapping_constant_across_frames": mapping_constant,
            "layout_positions": int(len(mapping_first)),
            "unique_raw_sensor_indices_in_first_frame": len(unique_indices),
            "raw_sensor_indices_first_frame": [int(x) for x in mapping_first.tolist()],
            "tactile_timestamps": {
                "strictly_increasing": bool(np.all(np.diff(timestamps) > 0)),
                "non_decreasing": bool(np.all(np.diff(timestamps) >= 0)),
                "positive_interval_count": int(np.sum(np.diff(timestamps) > 0)),
                "duplicate_interval_count": int(np.sum(np.diff(timestamps) == 0)),
                "negative_interval_count": int(np.sum(np.diff(timestamps) < 0)),
                "duplicate_pressure_rows_identical_to_previous": duplicate_pressure_equal_count,
                "duplicate_pressure_rows_different_from_previous": int(duplicate_mask.sum()) - duplicate_pressure_equal_count,
                "median_positive_interval_seconds": float(np.median(np.diff(timestamps)[np.diff(timestamps) > 0])),
                "interval_p01_seconds": float(np.percentile(np.diff(timestamps), 1)),
                "interval_p99_seconds": float(np.percentile(np.diff(timestamps), 99)),
            },
            "frame_timestamps": {
                "strictly_increasing": bool(np.all(np.diff(frame_timestamps) > 0)),
                "duplicate_interval_count": int(np.sum(np.diff(frame_timestamps) == 0)),
                "negative_interval_count": int(np.sum(np.diff(frame_timestamps) < 0)),
                "median_positive_interval_seconds": float(np.median(np.diff(frame_timestamps)[np.diff(frame_timestamps) > 0])),
            },
            "pressure_statistics_162": pressure_stats,
            "episode_counts": {
                "gloved_hand_only": sum(x["gloved_hand_only"] for x in episodes),
                "direct_pairing_only": sum(x["direct_pairing_only"] for x in episodes),
                "gloved_hand_direct": sum(x["gloved_hand_only"] and x["direct_pairing_only"] for x in episodes),
                "timestamp_integrity_failures": sum(not x["timestamp_integrity_ok"] for x in episodes),
                "preliminary_3s_window_eligible": len(eligible),
            },
            "eligible_trajectory_counts_by_participant": participant_counts,
            "episodes": episodes,
            "video_decoded": False,
            "limitations": [
                "Only pressure, timestamps, and permitted metadata arrays were read; RGB arrays were not accessed.",
                "3-second duration eligibility is preliminary and is not a final contact-window eligibility decision.",
                "The literal pairing_quality value 'direct' is checked explicitly; any other value remains excluded pending schema review.",
                "Regional mapping and contact-onset rules are not yet frozen by this script.",
            ],
        }
    finally:
        store.close()


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--object", required=True, dest="object_name")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_oxt_zarr_archive(args.archive, args.object_name)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "object_name": result["object_name"],
        "frame_count": result["frame_count"],
        "episode_count": result["episode_count"],
        "preliminary_3s_window_eligible": result["episode_counts"]["preliminary_3s_window_eligible"],
        "unique_raw_sensor_indices": result["unique_raw_sensor_indices_in_first_frame"],
    }, indent=2))


if __name__ == "__main__":
    main()
