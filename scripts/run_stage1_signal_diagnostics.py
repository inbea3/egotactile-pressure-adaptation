#!/usr/bin/env python3
"""Signal-only Stage 1 diagnostics on a metadata-stratified EgoTactile object sample.

No models are trained. Only pressure, tactile timestamps, episode boundaries and
permitted metadata are read. RGB/video arrays are never accessed.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import zarr
from huggingface_hub import hf_hub_download

try:
    from .build_sensor_mapping import build_sensor_mapping
except ImportError:
    from build_sensor_mapping import build_sensor_mapping

REPO_ID = "HustleHard/EgoTactile-OXT"
REVISION = "5b05f19e613758e76c330671f4c9f3c06af8af44"
DEFAULT_OBJECTS = (
    "Apple,CannedLuncheonMeat,BodyWash,SprayBottle,Clip,TapeMeasure,"
    "Dumbbell,TennisBall,ElectricToothbrush,PowerBank-20000mAh,GlassBowl,Mug"
)


def deduplicate_episode(ts: np.ndarray, pressure: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    if len(ts) != len(pressure):
        raise ValueError("timestamp/pressure length mismatch")
    keep = np.ones(len(ts), dtype=bool)
    conflict_count = 0
    negative_count = 0
    for i in range(1, len(ts)):
        delta = ts[i] - ts[i - 1]
        if delta < 0:
            negative_count += 1
        elif delta == 0:
            if np.array_equal(pressure[i], pressure[i - 1]):
                keep[i] = False
            else:
                conflict_count += 1
    valid = negative_count == 0 and conflict_count == 0
    return ts[keep], pressure[keep], {
        "negative_interval_count": negative_count,
        "duplicate_rows_removed": int((~keep).sum()),
        "duplicate_pressure_conflict_count": conflict_count,
        "timestamp_pressure_integrity_ok": valid,
    }


def find_sustained_onset(ts: np.ndarray, condition: np.ndarray, run_length: int = 3, max_gap: float = 0.5) -> int | None:
    if len(ts) != len(condition):
        raise ValueError("condition/timestamp length mismatch")
    for start in range(0, len(condition) - run_length + 1):
        stop = start + run_length
        if not bool(np.all(condition[start:stop])):
            continue
        if bool(np.all(np.diff(ts[start:stop]) > 0)) and bool(np.all(np.diff(ts[start:stop]) <= max_gap)):
            return start
    return None




def detect_contact_segments(
    ts: np.ndarray,
    high_condition: np.ndarray,
    low_condition: np.ndarray,
    start_run: int = 3,
    release_run: int = 3,
    max_gap: float = 0.5,
) -> list[dict]:
    """Hysteresis segmentation; exploratory only, not a frozen detector."""
    segments = []
    state = "outside"
    start_index = None
    high_count = 0
    low_count = 0
    for i in range(len(ts)):
        if i > 0 and ts[i] - ts[i - 1] > max_gap:
            if state == "inside" and start_index is not None:
                segments.append({
                    "start_index": start_index,
                    "end_index_exclusive": i,
                    "duration_seconds": float(ts[i - 1] - ts[start_index]),
                    "ended_by_timestamp_gap": True,
                })
            state = "outside"
            start_index = None
            high_count = 0
            low_count = 0

        if state == "outside":
            high_count = high_count + 1 if bool(high_condition[i]) else 0
            if high_count >= start_run:
                start_index = i - start_run + 1
                state = "inside"
                low_count = 0
        else:
            low_count = low_count + 1 if bool(low_condition[i]) else 0
            if low_count >= release_run:
                end_index = i - release_run + 1
                if start_index is not None and end_index > start_index:
                    segments.append({
                        "start_index": start_index,
                        "end_index_exclusive": end_index,
                        "duration_seconds": float(ts[end_index - 1] - ts[start_index]),
                        "ended_by_timestamp_gap": False,
                    })
                state = "outside"
                start_index = None
                high_count = 0
                low_count = 0

    if state == "inside" and start_index is not None:
        segments.append({
            "start_index": start_index,
            "end_index_exclusive": len(ts),
            "duration_seconds": float(ts[-1] - ts[start_index]),
            "ended_by_timestamp_gap": False,
        })
    return segments


def _strings(array) -> np.ndarray:
    return np.char.strip(np.asarray(array[:]).astype(str))


def _window_counts(ts: np.ndarray, onset_index: int) -> dict:
    onset = float(ts[onset_index])
    early_end = onset + 1.0
    future_end = onset + 3.0
    early = (ts >= onset) & (ts < early_end)
    future = (ts >= early_end) & (ts < future_end)
    return {
        "onset_seconds_from_episode_start": onset - float(ts[0]),
        "early_sample_count": int(early.sum()),
        "future_sample_count": int(future.sum()),
        "early_window_covered": bool(ts[0] <= onset and ts[-1] >= early_end),
        "future_window_covered": bool(ts[-1] >= future_end),
        "early_future_windows_disjoint": True,
    }


def _physical_and_region_loads(pressure_162: np.ndarray, mapping: dict) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    physical_columns = []
    physical_regions = []
    for sensor in mapping["physical_sensors"]:
        channels = sensor["layout_channels"]
        if sensor["multiplicity"] == 6:
            values = pressure_162[:, channels].sum(axis=1)
        else:
            values = pressure_162[:, channels[0]]
        physical_columns.append(values)
        physical_regions.append(sensor["region"])
    physical = np.column_stack(physical_columns)
    region_loads = {}
    for region in ("thumb", "index", "middle", "ring", "little", "palm"):
        indices = [i for i, value in enumerate(physical_regions) if value == region]
        region_loads[region] = physical[:, indices].sum(axis=1)
    return physical, region_loads


def _audit_object(object_name: str, archive_path: Path, mapping: dict) -> dict:
    store = zarr.ZipStore(str(archive_path), mode="r")
    try:
        root = zarr.open_group(store=store, path=f"{object_name}.zarr", mode="r")
        data = root["data"]
        ends = np.asarray(root["meta"]["episode_ends"][:], dtype=np.int64)
        pressure = np.asarray(data["right_tactile_data_pressure"][:], dtype=np.float32)
        # `timestamps` is the synchronized master-clock timeline for the pressure sequence.
        # `tactile_timestamps` records source glove update times and may repeat when a
        # previous glove readout is held at a later master tick.
        master_ts = np.asarray(data["timestamps"][:], dtype=np.float64)
        tactile_ts = np.asarray(data["tactile_timestamps"][:], dtype=np.float64)
        acquisition = _strings(data["acquisition_mode"])
        pairing = _strings(data["pairing_quality"])
        subjects = _strings(data["subject_id"])
        categories = _strings(data["object_category"])
        materials = _strings(data["object_surface_material"])
        fill_states = _strings(data["object_fill_state"])

        if pressure.shape[1] != 162 or pressure.shape[0] != len(master_ts) or len(tactile_ts) != len(master_ts):
            raise ValueError(f"Unexpected pressure/time shape for {object_name}")

        rows = []
        start = 0
        for ep_index, end_value in enumerate(ends.tolist()):
            end = int(end_value)
            acq_values = sorted(set(acquisition[start:end].tolist()))
            pair_values = sorted(set(pairing[start:end].tolist()))
            subject_values = sorted(set(subjects[start:end].tolist()))
            category_values = sorted(set(categories[start:end].tolist()))
            material_values = sorted(set(materials[start:end].tolist()))
            fill_values = sorted(set(fill_states[start:end].tolist()))
            if acq_values != ["gloved_hand"] or pair_values != ["direct"] or len(subject_values) != 1:
                start = end
                continue

            ts = master_ts[start:end]
            p162 = pressure[start:end]
            master_dt = np.diff(ts)
            source_tactile_ts = tactile_ts[start:end]
            source_tactile_dt = np.diff(source_tactile_ts)
            source_duplicate_mask = source_tactile_dt == 0
            source_duplicate_pressure_mismatch = 0
            if source_duplicate_mask.any():
                source_duplicate_pressure_mismatch = int(np.sum(np.any(
                    p162[1:][source_duplicate_mask] != p162[:-1][source_duplicate_mask], axis=1
                )))
            time_qc = {
                "negative_master_interval_count": int(np.sum(master_dt < 0)),
                "duplicate_master_interval_count": int(np.sum(master_dt == 0)),
                "master_timestamp_finite": bool(np.isfinite(ts).all()),
                "tactile_source_duplicate_interval_count": int(source_duplicate_mask.sum()),
                "tactile_source_duplicate_pressure_mismatch_count": source_duplicate_pressure_mismatch,
                "timestamp_pressure_integrity_ok": bool(
                    np.isfinite(ts).all() and np.all(master_dt > 0) and np.isfinite(p162).all()
                ),
            }
            row = {
                "episode_index": ep_index,
                "subject_id": subject_values[0],
                "category_values": category_values,
                "surface_material_values": material_values,
                "fill_state_values": fill_values,
                "raw_frame_count": end - start,
                **time_qc,
            }
            start = end
            if not time_qc["timestamp_pressure_integrity_ok"] or len(ts) < 2:
                row["excluded_from_signal_diagnostics"] = True
                rows.append(row)
                continue

            physical, regions = _physical_and_region_loads(p162, mapping)
            # Apply the paper's 5 N per-sensor threshold as a diagnostic candidate only.
            active = physical > 5.0
            total_raw = physical.sum(axis=1)
            total_excess = np.maximum(physical - 5.0, 0.0).sum(axis=1)
            active_count = active.sum(axis=1)
            conditions = {
                "any_sensor_gt_5N_3samples": active_count >= 1,
                "two_sensors_gt_5N_3samples": active_count >= 2,
                "total_load_gt_5N_3samples": total_raw > 5.0,
                "sum_excess_over_5N_gt_5N_3samples": total_excess > 5.0,
            }
            candidates = {}
            cycle_conditions = {
                "any_sensor_gt_5N_3samples": (active_count >= 1, active_count == 0),
                "two_sensors_gt_5N_3samples": (active_count >= 2, active_count < 2),
                "total_load_gt_5N_3samples": (total_raw > 5.0, total_raw < 2.5),
                "sum_excess_over_5N_gt_5N_3samples": (total_excess > 5.0, total_excess < 2.5),
            }
            for name, condition in conditions.items():
                onset_index = find_sustained_onset(ts, condition, run_length=3, max_gap=0.5)
                high_condition, low_condition = cycle_conditions[name]
                segments = detect_contact_segments(ts, high_condition, low_condition, max_gap=0.5)
                segment_windows = []
                for segment in segments:
                    start_idx = segment["start_index"]
                    segment_windows.append({
                        **segment,
                        "early_future_window_covered": bool(
                            ts[segment["end_index_exclusive"] - 1] >= ts[start_idx] + 3.0
                        ),
                    })
                candidates[name] = {
                    "first_onset": None if onset_index is None else _window_counts(ts, onset_index),
                    "detected_contact_segments": segment_windows,
                    "contact_segment_count": len(segment_windows),
                    "median_segment_duration_seconds": (
                        float(np.median([x["duration_seconds"] for x in segment_windows]))
                        if segment_windows else None
                    ),
                }

            positive_dt = np.diff(ts)
            relative_ts = ts - ts[0]
            row.update({
                "excluded_from_signal_diagnostics": False,
                "master_sample_count": int(len(ts)),
                "episode_duration_seconds": float(ts[-1] - ts[0]),
                "median_master_interval_seconds": float(np.median(positive_dt)),
                "max_master_interval_seconds": float(np.max(positive_dt)),
                "pressure_finite_fraction": float(np.isfinite(p162).mean()),
                "physical_sensor_count": int(physical.shape[1]),
                "median_active_physical_sensors_gt_5N": float(np.median(active_count)),
                "p95_active_physical_sensors_gt_5N": float(np.percentile(active_count, 95)),
                "total_load_p50_N": float(np.percentile(total_raw, 50)),
                "total_load_p95_N": float(np.percentile(total_raw, 95)),
                "total_load_p99_N": float(np.percentile(total_raw, 99)),
                "candidate_onsets": candidates,
                "region_load_p95_N": {
                    key: float(np.percentile(value, 95)) for key, value in regions.items()
                },
                "time_binned_pressure_trace_0p5s": [
                    {
                        "time_bin_start_seconds": float(left),
                        "total_load_mean_N": float(total_raw[(relative_ts >= left) & (relative_ts < left + 0.5)].mean()) if np.any((relative_ts >= left) & (relative_ts < left + 0.5)) else None,
                        "active_sensor_count_mean_gt_5N": float(active_count[(relative_ts >= left) & (relative_ts < left + 0.5)].mean()) if np.any((relative_ts >= left) & (relative_ts < left + 0.5)) else None,
                        "region_load_mean_N": {
                            key: float(value[(relative_ts >= left) & (relative_ts < left + 0.5)].mean()) if np.any((relative_ts >= left) & (relative_ts < left + 0.5)) else None
                            for key, value in regions.items()
                        },
                    }
                    for left in np.arange(0.0, float(ts[-1] - ts[0]), 0.5)
                ],
            })
            rows.append(row)

        return {
            "object_name": object_name,
            "frame_count": int(pressure.shape[0]),
            "episode_count": int(len(ends)),
            "metadata_category_values": sorted(set(categories.tolist())),
            "direct_gloved_episodes_audited": len(rows),
            "episodes": rows,
        }
    finally:
        store.close()


def run_diagnostics(objects: list[str], cache_dir: Path, output: Path) -> dict:
    note_path = Path(hf_hub_download(
        repo_id=REPO_ID, repo_type="dataset",
        filename="supplementary_tactile_note.json",
        revision=REVISION, cache_dir=cache_dir,
    ))
    mapping = build_sensor_mapping({
        "revision": REVISION,
        "raw_mapping_note": json.loads(note_path.read_text(encoding="utf-8")),
    })
    results = []
    for i, object_name in enumerate(objects, start=1):
        if not object_name or "/" in object_name or "\\" in object_name or ".." in object_name:
            raise ValueError(f"Unsafe object name: {object_name!r}")
        archive = Path(hf_hub_download(
            repo_id=REPO_ID, repo_type="dataset",
            filename=f"EgoTactile/{object_name}.zip",
            revision=REVISION, cache_dir=cache_dir,
        ))
        result = _audit_object(object_name, archive, mapping)
        results.append(result)
        print(json.dumps({
            "progress": f"{i}/{len(objects)}",
            "object": object_name,
            "category": result["metadata_category_values"],
            "direct_gloved_episodes": result["direct_gloved_episodes_audited"],
        }), flush=True)

    all_episodes = [ep for obj in results for ep in obj["episodes"]]
    valid = [ep for ep in all_episodes if not ep["excluded_from_signal_diagnostics"]]
    candidate_names = [
        "any_sensor_gt_5N_3samples",
        "two_sensors_gt_5N_3samples",
        "total_load_gt_5N_3samples",
        "sum_excess_over_5N_gt_5N_3samples",
    ]
    candidate_summary = {}
    for name in candidate_names:
        detected = [ep for ep in valid if ep["candidate_onsets"][name]["first_onset"] is not None]
        cycle_counts = [ep["candidate_onsets"][name]["contact_segment_count"] for ep in valid]
        covered = [
            ep for ep in detected
            if ep["candidate_onsets"][name]["first_onset"]["early_window_covered"]
            and ep["candidate_onsets"][name]["first_onset"]["future_window_covered"]
        ]
        candidate_summary[name] = {
            "onset_detected_episodes": len(detected),
            "early_and_future_windows_covered": len(covered),
            "onset_time_median_seconds": (
                float(np.median([ep["candidate_onsets"][name]["first_onset"]["onset_seconds_from_episode_start"] for ep in detected]))
                if detected else None
            ),
            "contact_segments_per_recording_median": float(np.median(cycle_counts)) if cycle_counts else None,
            "recordings_with_exactly_five_segments": sum(x == 5 for x in cycle_counts),
        }

    report = {
        "dataset": REPO_ID,
        "revision": REVISION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "objects_requested": objects,
        "object_count": len(results),
        "episodes_direct_gloved_audited": len(all_episodes),
        "episodes_passing_timestamp_integrity": len(valid),
        "object_categories": sorted({cat for x in results for cat in x["metadata_category_values"]}),
        "candidate_onset_summary": candidate_summary,
        "paper_repetition_protocol": {
            "repetitions_per_participant_object_recording": 5,
            "approximate_seconds_per_repetition": 6,
            "approximate_seconds_per_recording": 30,
            "source": "EgoTactile paper Appendix A.1.1 Object Inventory and Attributes / acquisition protocol",
        },
        "paper_threshold_reference": {
            "per_sensor_noise_threshold_N": 5.0,
            "normalization_reference_N": 200.0,
            "interpretation": "5 N is a per-sensor noise-filter candidate from the paper, not a validated whole-hand contact-onset threshold.",
        },
        "sensor_mapping": {
            "layout_positions": mapping["layout_positions"],
            "unique_physical_sensors": mapping["unique_physical_sensors"],
            "regions": [
                {k: x[k] for k in ("region", "unique_physical_sensor_count")}
                for x in mapping["regions"]
            ],
        },
        "objects": results,
        "video_decoded": False,
        "model_training_performed": False,
        "limitations": [
            "One OXT episode is a participant-object recording that contains multiple repeated grasp cycles; it is not one single grasp repetition.",
            "Exploratory signal-only diagnostics; onset candidates and cycle segmentation are not frozen.",
            "No model performance was examined or used to select thresholds.",
            "Only directly paired gloved-hand trajectories are included.",
            "Pressure and timestamps only; RGB/video arrays are not read.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objects", default=DEFAULT_OBJECTS)
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/egotactile_stage1_cache"))
    parser.add_argument("--output", type=Path, default=Path("results/egotactile_stage1_signal_diagnostics.json"))
    args = parser.parse_args()
    objects = [x.strip() for x in args.objects.split(",") if x.strip()]
    report = run_diagnostics(objects, args.cache_dir, args.output)
    print(json.dumps({
        "objects": report["object_count"],
        "categories": report["object_categories"],
        "episodes": report["episodes_direct_gloved_audited"],
        "timestamp_valid_episodes": report["episodes_passing_timestamp_integrity"],
        "candidate_onsets": report["candidate_onset_summary"],
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
