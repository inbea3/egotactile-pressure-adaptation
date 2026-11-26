#!/usr/bin/env python3
"""Run Stage 1 pressure-event and six-region target inventory on all 63 objects."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi

try:
    from .run_stage1_signal_diagnostics import run_diagnostics
except ImportError:
    from run_stage1_signal_diagnostics import run_diagnostics

REPO_ID = "HustleHard/EgoTactile-OXT"
REVISION = "5b05f19e613758e76c330671f4c9f3c06af8af44"


def discover_object_names() -> list[str]:
    api = HfApi()
    tree = api.list_repo_tree(
        REPO_ID, repo_type="dataset", revision=REVISION, recursive=True
    )
    names = sorted(
        Path(item.path).stem
        for item in tree
        if getattr(item, "path", "").startswith("EgoTactile/")
        and getattr(item, "path", "").lower().endswith(".zip")
    )
    if len(names) != 63:
        raise RuntimeError(f"Expected 63 object archives, found {len(names)}")
    return names


def run_full_stage1(cache_dir: Path, output: Path) -> dict:
    objects = discover_object_names()
    report = run_diagnostics(objects, cache_dir, output)
    report["full_object_inventory"] = True
    report["object_name_manifest_source"] = f"{REPO_ID}@{REVISION}/EgoTactile/*.zip"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/egotactile_stage1_full_cache"))
    parser.add_argument("--output", type=Path, default=Path("results/egotactile_stage1_full_event_inventory.json"))
    args = parser.parse_args()
    report = run_full_stage1(args.cache_dir, args.output)
    cycle = report["grasp_cycle_detection_sensitivity"]
    print(json.dumps({
        "objects": report["object_count"],
        "recordings": report["episodes_direct_gloved_audited"],
        "master_timestamp_valid_recordings": report["episodes_passing_timestamp_integrity"],
        "categories": report["object_categories"],
        "cycle_detection": cycle,
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
