#!/usr/bin/env python3
"""Create a metadata-only manifest for the public EgoTactile-OXT Hub.

This script queries the Hugging Face repository tree and records object ZIP paths,
file sizes, and revision metadata. It does not download ZIP archives or read data.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi


REPO_ID = "HustleHard/EgoTactile-OXT"


def build_manifest(revision: str = "main") -> dict:
    api = HfApi()
    info = api.dataset_info(REPO_ID, revision=revision, files_metadata=True)
    paths = api.list_repo_tree(
        REPO_ID, repo_type="dataset", revision=info.sha, recursive=True
    )
    files = []
    for item in paths:
        path = getattr(item, "path", "")
        if not path:
            continue
        size = getattr(item, "size", None)
        lfs = getattr(item, "lfs", None)
        if path.lower().endswith(".zip") or path.endswith("supplementary_tactile_note.json"):
            files.append({
                "path": path,
                "size_bytes": size,
                "lfs": (
                    {
                        "sha256": getattr(lfs, "sha256", None),
                        "size": getattr(lfs, "size", None),
                    } if lfs else None
                ),
                "kind": "object_archive" if path.lower().endswith(".zip") else "mapping_note",
            })
    files.sort(key=lambda x: x["path"])
    archives = [x for x in files if x["kind"] == "object_archive"]
    return {
        "dataset": REPO_ID,
        "revision": info.sha,
        "requested_revision": revision,
        "license": "cc-by-nc-4.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "object_archive_count": len(archives),
        "object_archives_total_bytes": sum(x["size_bytes"] or 0 for x in archives),
        "files": files,
        "downloaded_any_dataset_archive": False,
        "notes": [
            "Metadata-only Hub listing; no archive contents were downloaded.",
            "File names and sizes are not a substitute for trajectory-level counts.",
            "The resolved dataset commit SHA pins the source revision.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--output", type=Path, default=Path("results/egotactile_hub_manifest.json"))
    args = parser.parse_args()
    manifest = build_manifest(args.revision)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "dataset": manifest["dataset"],
        "revision": manifest["revision"],
        "object_archive_count": manifest["object_archive_count"],
        "object_archives_total_bytes": manifest["object_archives_total_bytes"],
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
