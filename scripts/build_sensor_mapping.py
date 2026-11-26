#!/usr/bin/env python3
"""Build an explicit 162-layout -> 137-physical-sensor -> 6-region mapping."""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


AREA_TO_REGION = {
    "thumb_tip": "thumb", "thumb_bend": "thumb",
    "index_tip": "index", "index_bend": "index",
    "middle_tip": "middle", "middle_bend": "middle",
    "ring_tip": "ring", "ring_bend": "ring",
    "little_tip": "little", "little_bend": "little",
    "palm": "palm",
}


def _unwrap_note(value: dict) -> dict:
    if "raw_mapping_note" in value:
        value = value["raw_mapping_note"]
    if "EgoTactilePressureGlove" in value:
        value = value["EgoTactilePressureGlove"]
    return value


def build_sensor_mapping(note_wrapper: dict, expected_layout: int = 162, expected_unique: int = 137) -> dict:
    note = _unwrap_note(note_wrapper)
    area_names = note["functional_area_idx_description"]
    descriptions = note["sensor_idx_description"]
    rows = []
    groups: dict[int, list[int]] = defaultdict(list)

    for layout_channel in range(int(note["sensor_num"])):
        description = descriptions[f"sensor_idx_{layout_channel}"]
        area_match = re.search(r"functional_area_idx=(\d+)", description)
        grid_match = re.search(r"grid \(y=(\d+), x=(\d+)\)", description)
        raw_match = re.search(r"source raw pressure index=(\d+) \(1-based\)", description)
        if not (area_match and grid_match and raw_match):
            raise ValueError(f"Cannot parse official mapping description for channel {layout_channel}")
        area_idx = int(area_match.group(1))
        area = area_names[str(area_idx)]
        raw_1based = int(raw_match.group(1))
        if area not in AREA_TO_REGION:
            raise ValueError(f"Unrecognized functional area: {area}")
        row = {
            "layout_channel": layout_channel,
            "functional_area_idx": area_idx,
            "functional_area": area,
            "region": AREA_TO_REGION[area],
            "grid_y": int(grid_match.group(1)),
            "grid_x": int(grid_match.group(2)),
            "source_raw_index_1based": raw_1based,
            "source_raw_index_0based": raw_1based - 1,
        }
        rows.append(row)
        groups[raw_1based].append(layout_channel)

    if len(rows) != expected_layout:
        raise ValueError(f"Expected {expected_layout} layout channels, found {len(rows)}")
    if len(groups) != expected_unique:
        raise ValueError(f"Expected {expected_unique} unique physical sensors, found {len(groups)}")

    physical = []
    for raw_1based, channels in sorted(groups.items()):
        group_rows = [rows[i] for i in channels]
        areas = {r["functional_area"] for r in group_rows}
        regions = {r["region"] for r in group_rows}
        if len(areas) != 1 or len(regions) != 1:
            raise ValueError(f"Raw sensor {raw_1based} maps to multiple functional areas")
        area = group_rows[0]["functional_area"]
        multiplicity = len(channels)
        if multiplicity not in (1, 6):
            raise ValueError(f"Unexpected layout multiplicity {multiplicity} for raw index {raw_1based}")
        if multiplicity == 6 and not area.endswith("_bend"):
            raise ValueError(f"Only bend sensors may be expanded into six virtual cells: {raw_1based}")
        for row in group_rows:
            row["layout_multiplicity"] = multiplicity
            row["aggregation_rule"] = "sum_virtual_cells" if multiplicity == 6 else "single_cell"
            row["virtual_channels"] = channels
        physical.append({
            "source_raw_index_1based": raw_1based,
            "source_raw_index_0based": raw_1based - 1,
            "functional_area": area,
            "region": group_rows[0]["region"],
            "layout_channels": channels,
            "multiplicity": multiplicity,
            "aggregation_rule": "sum_virtual_cells" if multiplicity == 6 else "single_cell",
        })

    multiplicities = {n: sum(x["multiplicity"] == n for x in physical) for n in (1, 6)}
    if expected_layout == 162 and expected_unique == 137 and multiplicities != {1: 132, 6: 5}:
        raise ValueError(f"Unexpected multiplicity distribution: {multiplicities}")

    regions = []
    for region in ("thumb", "index", "middle", "ring", "little", "palm"):
        selected = [x for x in physical if x["region"] == region]
        layout_channels = sum(len(x["layout_channels"]) for x in selected)
        regions.append({
            "region": region,
            "functional_areas": sorted({x["functional_area"] for x in selected}),
            "unique_physical_sensor_count": len(selected),
            "layout_channel_count": layout_channels,
            "physical_sensor_raw_indices_1based": [x["source_raw_index_1based"] for x in selected],
        })

    return {
        "dataset": "HustleHard/EgoTactile-OXT",
        "revision": note_wrapper.get("revision"),
        "source_note_sha256": note_wrapper.get("sha256"),
        "layout_positions": len(rows),
        "unique_physical_sensors": len(physical),
        "raw_index_convention": "OXT right_tactile_raw_index_pressure matches note's 1-based source indices; subtract one only when indexing Python raw_pressure_256.",
        "multiplicity_distribution": multiplicities,
        "bend_expansion_rule": "Each of five physical bend sensors is represented by six virtual layout cells, each equal to one sixth of the source sensor value. Sum the six cells to reconstruct the physical sensor value.",
        "regions": regions,
        "physical_sensors": physical,
        "layout_channels": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Mapping-note audit wrapper JSON")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    wrapper = json.loads(args.input.read_text(encoding="utf-8"))
    result = build_sensor_mapping(wrapper)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "layout_positions": result["layout_positions"],
        "unique_physical_sensors": result["unique_physical_sensors"],
        "multiplicity_distribution": result["multiplicity_distribution"],
        "regions": [
            {k: x[k] for k in ("region", "unique_physical_sensor_count", "layout_channel_count")}
            for x in result["regions"]
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
