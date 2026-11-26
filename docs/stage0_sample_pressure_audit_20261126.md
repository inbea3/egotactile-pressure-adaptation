# Stage 0 Sample Pressure Audit — 3 Objects

- Date: 2026-11-26
- Dataset revision: `5b05f19e613758e76c330671f4c9f3c06af8af44`
- Workflow: https://github.com/inbea3/egotactile-pressure-adaptation/actions/runs/45800726767
- Artifact ID: `15107490387`
- Artifact SHA-256: `f81677fe58031c5ab330636c29ec8a9f3707230c6d128542373e8859becffe6b`
- Downloaded object archives: 3
- Total downloaded bytes: 171,224,762
- No video/RGB arrays decoded.

## 1. Sample results

| Object | Frames | Episodes | gloved_hand + direct | Preliminary duration/signal eligible | Distinct participants in eligible episodes |
|---|---:|---:|---:|---:|---:|
| Apple | 5,632 | 14 | 11 | 11 | 11 |
| SprayBottle | 4,855 | 11 | 11 | 11 | 11 |
| TennisBall | 5,715 | 14 | 11 | 11 | 11 |

The other three episodes in Apple and TennisBall are weakly paired bare-hand records and are excluded from the primary analysis. SprayBottle contains 11 gloved-hand direct episodes.

The preliminary eligibility check requires gloved-hand acquisition, literal `pairing_quality == direct`, finite 162-layout pressure values, non-negative within-episode tactile timestamp intervals, and at least 3 seconds of elapsed time. This is only a duration/signal gate; contact onset and final window eligibility have not yet been computed.

## 2. Sensor mapping validation

For all three objects:
- `right_tactile_data_pressure`: shape `[N, 162]`, float32;
- `right_tactile_raw_pressure_256`: shape `[N, 256]`, uint16;
- `right_tactile_raw_index_pressure`: shape `[N, 162]`;
- raw-index mapping is constant across all frames in each object archive;
- first-frame raw-index mapping contains exactly **137 unique physical sensor indices**.

This empirically confirms the 162 layout / 137 unique index distinction in the three sampled object archives.

## 3. Pressure signal integrity

| Object | Finite fraction | Min | Max | P99 |
|---|---:|---:|---:|---:|
| Apple | 1.000 | 0 N | 119 N | 41 N |
| SprayBottle | 1.000 | 0 N | 129 N | 32 N |
| TennisBall | 1.000 | 0 N | 92 N | 25 N |

These are descriptive statistics of the released `right_tactile_data_pressure` array, not calibrated force validation.

## 4. Timestamp issue requiring resolution

The tactile timestamp arrays contain many repeated adjacent values. The sample archives also show one negative interval at object-concatenated level for Apple and TennisBall, but no negative intervals within individual episode slices in the sampled records. The frame-level `timestamps` arrays have no duplicates in these samples and show one negative boundary interval at concatenated object level for Apple and TennisBall.

This likely reflects asynchronous/held pressure samples and/or episode concatenation behavior, but that interpretation is not yet proven. Before freezing time-based windows:
1. compare adjacent pressure vectors whenever tactile timestamps are equal;
2. quantify whether repeated timestamp rows carry identical pressure values;
3. inspect each episode independently and separate true within-episode reversals from cross-episode resets;
4. compare tactile timestamps with frame timestamps and verify units/rate;
5. choose a documented deduplication or zero-duration handling rule without inventing time samples.

Do not simply sort all samples globally; that could destroy episode order. Do not use frame timestamps as a substitute for tactile timestamps without documenting the synchronization semantics.

## 5. Stage 0 decision

**Sample-level GO** for continuing to the full 63-object inventory:
- archive structure is readable;
- the required pressure arrays and episode boundaries exist;
- 137 unique physical indices are consistently represented;
- sampled objects each have 11 directly paired gloved-hand trajectories, sufficient for 5-shot plus one Query before final window exclusions.

**Not yet approved**:
- global 5-shot feasibility across all 63 objects;
- final time-window eligibility;
- final contact-onset detector;
- target construction;
- any model training.

## 6. Integrity boundary

This is an actual data-read audit, not a model experiment. The report does not claim that future structural targets are repeatable or that Support correction helps. Those questions belong to Stage 1 and Stage 2 after timestamp semantics are resolved.
