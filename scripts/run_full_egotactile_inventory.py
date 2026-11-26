#!/usr/bin/env python3
"""Full 63-object EgoTactile-OXT pressure-only trajectory inventory.

Downloads object ZIPs sequentially from a pinned Hub revision, reads only pressure,
timestamps, episode boundaries and permitted metadata, and writes aggregate/per-object
summaries. RGB arrays are never accessed. Raw data is not included in the output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

try:
    from .audit_oxt_zarr import audit_oxt_zarr_archive
except ImportError:
    from audit_oxt_zarr import audit_oxt_zarr_archive


REPO_ID = "HustleHard/EgoTactile-OXT"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_full_inventory(output: Path, cache_dir: Path, revision: str = "main") -> dict:
    api = HfApi()
    info = api.dataset_info(REPO_ID, revision=revision, files_metadata=True)
    resolved_sha = info.sha
    tree = api.list_repo_tree(
        REPO_ID, repo_type="dataset", revision=resolved_sha, recursive=True
    )
    object_files = [
        item for item in tree
        if getattr(item, "path", "").startswith("EgoTactile/")
        and getattr(item, "path", "").lower().endswith(".zip")
    ]
    object_files.sort(key=lambda x: x.path)
    if len(object_files) != 63:
        raise RuntimeError(
            f"Expected 63 object archives at pinned revision {resolved_sha}, "
            f"found {len(object_files)}. Stop rather than silently changing scope."
        )

    rows = []
    for index, item in enumerate(object_files, start=1):
        object_name = Path(item.path).stem
        expected_size = getattr(item, "size", None)
        lfs = getattr(item, "lfs", None)
        expected_sha = getattr(lfs, "sha256", None) if lfs else None
        local_path = Path(hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=item.path,
            revision=resolved_sha,
            cache_dir=cache_dir,
        ))
        actual_size = local_path.stat().st_size
        actual_sha = sha256_file(local_path)
        if expected_size is not None and actual_size != expected_size:
            raise RuntimeError(f"Size mismatch for {item.path}")
        if expected_sha and actual_sha != expected_sha:
            raise RuntimeError(f"SHA-256 mismatch for {item.path}")

        audit = audit_oxt_zarr_archive(local_path, object_name)
        audit["dataset_path"] = item.path
        audit["dataset_archive_size_bytes"] = actual_size
        audit["dataset_archive_sha256"] = actual_sha
        rows.append(audit)
        print(json.dumps({
            "progress": f"{index}/63",
            "object": object_name,
            "episodes": audit["episode_count"],
            "gloved_direct": audit["episode_counts"]["gloved_hand_only"],
            "preliminary_3s_eligible": audit["episode_counts"]["preliminary_3s_window_eligible"],
        }), flush=True)

    object_rows = []
    for row in rows:
        eligible_eps = [
            ep for ep in row["episodes"] if ep["preliminary_3s_window_eligible"]
        ]
        object_rows.append({
            "object_name": row["object_name"],
            "frames": row["frame_count"],
            "episodes": row["episode_count"],
            "gloved_hand_direct_episodes": row["episode_counts"]["gloved_hand_only"],
            "preliminary_3s_eligible_trajectories": len(eligible_eps),
            "distinct_participants_preliminary_eligible": len(row["eligible_trajectory_counts_by_participant"]),
            "participants_preliminary_eligible": sorted(row["eligible_trajectory_counts_by_participant"]),
            "unique_physical_sensor_indices": row["unique_raw_sensor_indices_in_first_frame"],
            "sensor_mapping_constant": row["raw_index_mapping_constant_across_frames"],
            "tactile_timestamp_negative_intervals": row["tactile_timestamps"]["negative_interval_count"],
            "tactile_timestamp_duplicate_intervals": row["tactile_timestamps"]["duplicate_interval_count"],
            "duplicate_timestamp_pressure_rows_identical": row["tactile_timestamps"]["duplicate_pressure_rows_identical_to_previous"],
            "duplicate_timestamp_pressure_rows_different": row["tactile_timestamps"]["duplicate_pressure_rows_different_from_previous"],
            "all_pressure_finite": row["pressure_statistics_162"]["finite_fraction"] == 1.0,
            "archive_sha256": row["dataset_archive_sha256"],
        })

    shot_eligibility = {
        str(k): sum(x["preliminary_3s_eligible_trajectories"] >= k + 1 for x in object_rows)
        for k in (1, 2, 5)
    }
    report = {
        "dataset": REPO_ID,
        "revision": resolved_sha,
        "license": "cc-by-nc-4.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "object_archive_count": len(rows),
        "total_downloaded_archive_bytes": sum(x["dataset_archive_size_bytes"] for x in rows),
        "total_frames": sum(x["frame_count"] for x in rows),
        "total_episodes": sum(x["episode_count"] for x in rows),
        "total_gloved_hand_direct_episodes": sum(x["episode_counts"]["gloved_hand_only"] for x in rows),
        "total_preliminary_3s_eligible_trajectories": sum(x["episode_counts"]["preliminary_3s_window_eligible"] for x in rows),
        "objects_meeting_preliminary_shot_gate": shot_eligibility,
        "object_summary": object_rows,
        "object_details": rows,
        "video_decoded": False,
        "raw_pressure_exported": False,
        "eligibility_definition": (
            "gloved_hand only; pairing_quality exactly 'direct'; finite 162-layout pressure; "
            "within-episode tactile timestamps have no negative intervals; elapsed duration >= 3 s. "
            "This is preliminary and does not include contact-onset or target validity."
        ),
        "limitations": [
            "This is a full archive/trajectory inventory, not a model experiment.",
            "Repeated tactile timestamps require interpretation; duplicate pressure consistency is reported.",
            "No contact-onset detector, early/future target, or regional grouping has been frozen by this run.",
            "No model training or performance evaluation was performed.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--output", type=Path, default=Path("results/egotactile_full_stage0_inventory.json"))
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/egotactile_full_cache"))
    args = parser.parse_args()
    report = run_full_inventory(args.output, args.cache_dir, args.revision)
    print(json.dumps({
        "revision": report["revision"],
        "objects": report["object_archive_count"],
        "frames": report["total_frames"],
        "episodes": report["total_episodes"],
        "gloved_direct": report["total_gloved_hand_direct_episodes"],
        "preliminary_eligible": report["total_preliminary_3s_eligible_trajectories"],
        "shot_gate": report["objects_meeting_preliminary_shot_gate"],
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
