#!/usr/bin/env python3
"""Inventory one EgoTactile-OXT object ZIP without decoding video or pressure arrays.

This is a conservative first-pass inspector. It reads only small JSON metadata
members (for example Zarr .zarray/.zattrs files), records archive checksums and
file layout, and never extracts the archive. It deliberately does not infer
trajectory counts until the actual episode-boundary schema has been verified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any


VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv"}
JSON_SUFFIXES = {".json", ".zarray", ".zattrs", ".zgroup", ".zmetadata"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_json_read(archive: zipfile.ZipFile, member: str) -> tuple[Any | None, str | None]:
    try:
        return json.loads(archive.read(member).decode("utf-8")), None
    except Exception as exc:  # malformed metadata must be reported, not hidden
        return None, f"{type(exc).__name__}: {exc}"


def inspect_archive(path: Path) -> dict[str, Any]:
    report: dict[str, Any] = {
        "archive": path.name,
        "archive_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "zip_readable": False,
        "member_count": 0,
        "suffix_counts": {},
        "top_level_prefixes": [],
        "video_members_not_decoded": [],
        "zarr_arrays": [],
        "zarr_attributes": [],
        "json_metadata": [],
        "metadata_errors": [],
        "warning": (
            "Inventory only. No pressure values were read, no videos were decoded, "
            "and no trajectory counts or data-quality conclusions are inferred."
        ),
    }

    with zipfile.ZipFile(path, "r") as archive:
        infos = [info for info in archive.infolist() if not info.is_dir()]
        names = [info.filename for info in infos]
        report["zip_readable"] = archive.testzip() is None
        report["member_count"] = len(infos)
        report["suffix_counts"] = dict(Counter(Path(name).suffix.lower() or "<none>" for name in names))
        report["top_level_prefixes"] = sorted({
            name.split("/", 1)[0] for name in names if "/" in name
        })
        report["video_members_not_decoded"] = [
            name for name in names if Path(name).suffix.lower() in VIDEO_SUFFIXES
        ]

        for member in names:
            suffix = Path(member).suffix.lower()
            if suffix not in JSON_SUFFIXES and Path(member).name not in {".zarray", ".zattrs", ".zgroup"}:
                continue
            data, error = safe_json_read(archive, member)
            if error:
                report["metadata_errors"].append({"member": member, "error": error})
                continue
            if not isinstance(data, dict):
                report["json_metadata"].append({
                    "member": member,
                    "json_type": type(data).__name__,
                })
                continue

            if member.endswith(".zarray"):
                report["zarr_arrays"].append({
                    "member": member,
                    "shape": data.get("shape"),
                    "chunks": data.get("chunks"),
                    "dtype": data.get("dtype"),
                    "compressor": data.get("compressor"),
                    "filters": data.get("filters"),
                    "order": data.get("order"),
                })
            elif member.endswith(".zattrs"):
                attrs = {
                    key: value for key, value in data.items()
                    if key in {
                        "episode_ends", "meta", "metadata", "timestamp",
                        "timestamps", "frequency", "sampling_rate",
                    }
                }
                report["zarr_attributes"].append({
                    "member": member,
                    "keys": sorted(data.keys()),
                    "selected_values": attrs,
                })
            else:
                report["json_metadata"].append({
                    "member": member,
                    "keys": sorted(data.keys()),
                })

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="Path to one object-level OXT ZIP")
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    args = parser.parse_args()

    report = inspect_archive(args.archive)
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
