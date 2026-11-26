#!/usr/bin/env python3
"""Fetch and structurally summarize the official EgoTactile-OXT tactile mapping note."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "HustleHard/EgoTactile-OXT"
PINNED_REVISION = "5b05f19e613758e76c330671f4c9f3c06af8af44"
FILENAME = "supplementary_tactile_note.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize(value, path="$"):
    if isinstance(value, dict):
        return {
            "type": "object",
            "keys": list(value.keys()),
            "children": {k: summarize(v, f"{path}.{k}") for k, v in value.items()},
        }
    if isinstance(value, list):
        return {
            "type": "array",
            "length": len(value),
            "sample": value[:5],
        }
    return {"type": type(value).__name__, "value": value}


def inspect_mapping_note(cache_dir: Path, output: Path) -> dict:
    local_path = Path(hf_hub_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        filename=FILENAME,
        revision=PINNED_REVISION,
        cache_dir=cache_dir,
    ))
    raw = json.loads(local_path.read_text(encoding="utf-8"))
    report = {
        "dataset": REPO_ID,
        "revision": PINNED_REVISION,
        "file": FILENAME,
        "sha256": sha256_file(local_path),
        "size_bytes": local_path.stat().st_size,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "structure": summarize(raw),
        "raw_mapping_note": raw,
        "status": "retrieved_and_parsed; semantic validation pending",
        "limitations": [
            "This step retrieves the official mapping note but does not decide anatomical regions.",
            "Every index and sensor-type definition must be reconciled with sampled raw_index_pressure arrays.",
            "Do not infer anatomical labels from image layout alone.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/egotactile_mapping_cache"))
    parser.add_argument("--output", type=Path, default=Path("results/egotactile_mapping_note_audit.json"))
    args = parser.parse_args()
    report = inspect_mapping_note(args.cache_dir, args.output)
    print(json.dumps({
        "revision": report["revision"],
        "sha256": report["sha256"],
        "size_bytes": report["size_bytes"],
        "top_level_keys": report["structure"].get("keys", []),
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
