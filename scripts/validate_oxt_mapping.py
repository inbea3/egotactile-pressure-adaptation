#!/usr/bin/env python3
"""Cross-check the official tactile note against OXT Zarr sensor-index/grid arrays."""
from __future__ import annotations

import argparse
import json
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


def validate_mapping_arrays(mapping: dict, raw_indices: np.ndarray, grid_xy: np.ndarray, pressure: np.ndarray) -> dict:
    expected_raw = np.asarray(
        [x["source_raw_index_1based"] for x in mapping["layout_channels"]],
        dtype=np.int64,
    )
    expected_grid = np.asarray(
        [[x["grid_x"], x["grid_y"]] for x in mapping["layout_channels"]],
        dtype=np.int64,
    )
    if raw_indices.ndim != 2 or raw_indices.shape[1] != len(expected_raw):
        raise ValueError("raw_index_pressure shape does not match mapping layout")
    if grid_xy.shape != (raw_indices.shape[0], len(expected_raw), 2):
        raise ValueError("grid_xy_pressure shape does not match mapping layout")
    if pressure.shape != raw_indices.shape:
        raise ValueError("pressure array shape does not match mapping layout")

    raw_stable = bool(np.all(raw_indices == raw_indices[0][None, :]))
    grid_stable = bool(np.all(grid_xy == grid_xy[0][None, :, :]))
    raw_matches_note = bool(np.array_equal(raw_indices[0], expected_raw))
    grid_matches_note = bool(np.array_equal(grid_xy[0], expected_grid))

    bend_groups = [
        x["layout_channels"] for x in mapping["physical_sensors"]
        if x["multiplicity"] > 1
    ]
    bend_value_mismatch_count = 0
    for channels in bend_groups:
        first = pressure[:, channels[0]]
        for channel in channels[1:]:
            bend_value_mismatch_count += int(np.sum(pressure[:, channel] != first))

    return {
        "raw_index_mapping_stable_across_frames": raw_stable,
        "grid_mapping_stable_across_frames": grid_stable,
        "raw_indices_match_official_note": raw_matches_note,
        "grid_coordinates_match_official_note": grid_matches_note,
        "bend_physical_sensor_group_count": len(bend_groups),
        "bend_virtual_cell_pressure_mismatch_count": bend_value_mismatch_count,
        "all_checks_pass": bool(
            raw_stable and grid_stable and raw_matches_note and grid_matches_note
            and bend_value_mismatch_count == 0
        ),
    }


def validate_objects(objects: list[str], cache_dir: Path) -> dict:
    note_path = Path(hf_hub_download(
        repo_id=REPO_ID, repo_type="dataset",
        filename="supplementary_tactile_note.json",
        revision=REVISION, cache_dir=cache_dir,
    ))
    note = json.loads(note_path.read_text(encoding="utf-8"))
    mapping = build_sensor_mapping({
        "revision": REVISION,
        "raw_mapping_note": note,
    })
    results = []
    for object_name in objects:
        if not object_name or "/" in object_name or "\\" in object_name or ".." in object_name:
            raise ValueError(f"Unsafe object name: {object_name!r}")
        archive = Path(hf_hub_download(
            repo_id=REPO_ID, repo_type="dataset",
            filename=f"EgoTactile/{object_name}.zip",
            revision=REVISION, cache_dir=cache_dir,
        ))
        store = zarr.ZipStore(str(archive), mode="r")
        try:
            root = zarr.open_group(store=store, path=f"{object_name}.zarr", mode="r")
            data = root["data"]
            raw_indices = np.asarray(data["right_tactile_raw_index_pressure"][:], dtype=np.int64)
            grid_xy = np.asarray(data["right_tactile_grid_xy_pressure"][:], dtype=np.int64)
            pressure = np.asarray(data["right_tactile_data_pressure"][:], dtype=np.float32)
            check = validate_mapping_arrays(mapping, raw_indices, grid_xy, pressure)
            results.append({
                "object_name": object_name,
                "frame_count": int(pressure.shape[0]),
                "pressure_shape": list(pressure.shape),
                "raw_index_shape": list(raw_indices.shape),
                "grid_shape": list(grid_xy.shape),
                "pressure_transform": root.attrs.get("pressure_transform"),
                **check,
            })
        finally:
            store.close()
    return {
        "dataset": REPO_ID,
        "revision": REVISION,
        "layout_positions": mapping["layout_positions"],
        "unique_physical_sensors": mapping["unique_physical_sensors"],
        "regions": [
            {k: x[k] for k in ("region", "unique_physical_sensor_count", "layout_channel_count")}
            for x in mapping["regions"]
        ],
        "objects": results,
        "all_objects_pass": all(x["all_checks_pass"] for x in results),
        "video_decoded": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objects", default="Apple,SprayBottle,TennisBall")
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/egotactile_mapping_validation"))
    parser.add_argument("--output", type=Path, default=Path("results/egotactile_mapping_crosswalk_validation.json"))
    args = parser.parse_args()
    result = validate_objects([x.strip() for x in args.objects.split(",") if x.strip()], args.cache_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "objects": len(result["objects"]),
        "all_objects_pass": result["all_objects_pass"],
        "unique_physical_sensors": result["unique_physical_sensors"],
        "object_results": [
            {k: x[k] for k in ("object_name", "raw_index_mapping_stable_across_frames", "grid_mapping_stable_across_frames", "raw_indices_match_official_note", "grid_coordinates_match_official_note", "bend_virtual_cell_pressure_mismatch_count", "all_checks_pass")}
            for x in result["objects"]
        ],
        "output": str(args.output),
    }, indent=2))
    if not result["all_objects_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
