#!/usr/bin/env python3
"""Download a small, explicit set of EgoTactile-OXT object ZIPs and inventory metadata only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

try:
    from .inspect_oxt_archive import inspect_archive
    from .audit_oxt_zarr import audit_oxt_zarr_archive
except ImportError:  # direct script execution from the scripts directory
    from inspect_oxt_archive import inspect_archive
    from audit_oxt_zarr import audit_oxt_zarr_archive


REPO_ID = "HustleHard/EgoTactile-OXT"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_sample_audit(objects: list[str], output: Path, cache_dir: Path) -> dict:
    api = HfApi()
    info = api.dataset_info(REPO_ID, revision="main", files_metadata=True)
    revision = info.sha
    tree = api.list_repo_tree(REPO_ID, repo_type="dataset", revision=revision, recursive=True)
    files = {getattr(x, "path", ""): x for x in tree}
    results = []
    for object_name in objects:
        if not object_name or "/" in object_name or "\\" in object_name or ".." in object_name:
            raise ValueError(f"Unsafe object name: {object_name!r}")
        filename = f"EgoTactile/{object_name}.zip"
        if filename not in files:
            raise FileNotFoundError(f"Object archive not found at pinned revision {revision}: {filename}")
        metadata = files[filename]
        expected_size = getattr(metadata, "size", None)
        local_path = Path(hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=filename,
            revision=revision,
            cache_dir=cache_dir,
        ))
        actual_size = local_path.stat().st_size
        if expected_size is not None and actual_size != expected_size:
            raise RuntimeError(f"Size mismatch for {filename}: expected {expected_size}, got {actual_size}")
        inventory = inspect_archive(local_path)
        pressure_audit = audit_oxt_zarr_archive(local_path, object_name)
        inventory.update({
            "pressure_only_trajectory_audit": pressure_audit,
            "object_name": object_name,
            "dataset_revision": revision,
            "dataset_path": filename,
            "expected_size_bytes": expected_size,
            "downloaded_size_bytes": actual_size,
            "downloaded_sha256": sha256_file(local_path),
        })
        results.append(inventory)

    report = {
        "dataset": REPO_ID,
        "revision": revision,
        "objects_requested": objects,
        "objects_completed": len(results),
        "downloaded_total_bytes": sum(x["downloaded_size_bytes"] for x in results),
        "pressure_arrays_decoded": False,
        "video_decoded": False,
        "trajectory_counts_inferred": False,
        "results": results,
        "limitations": [
            "This is a schema and archive-layout smoke audit only.",
            "Zarr chunk payloads and pressure values were not decoded.",
            "Do not use this report to claim per-object eligible trajectory counts.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objects", default="Apple,SprayBottle,TennisBall")
    parser.add_argument("--output", type=Path, default=Path("results/egotactile_sample_archive_audit.json"))
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/egotactile_hf_cache"))
    args = parser.parse_args()
    objects = [x.strip() for x in args.objects.split(",") if x.strip()]
    report = run_sample_audit(objects, args.output, args.cache_dir)
    print(json.dumps({
        "dataset_revision": report["revision"],
        "objects_completed": report["objects_completed"],
        "downloaded_total_bytes": report["downloaded_total_bytes"],
        "pressure_arrays_decoded": report["pressure_arrays_decoded"],
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
