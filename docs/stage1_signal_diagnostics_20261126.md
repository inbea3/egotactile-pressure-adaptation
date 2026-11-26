# Stage 1 Signal-Only Diagnostics — 12-Object Sample (2026-11-26)

## Provenance

- Dataset: `HustleHard/EgoTactile-OXT`
- Pinned revision: `5b05f19e613758e76c330671f4c9f3c06af8af44`
- Workflow: https://github.com/inbea3/egotactile-pressure-adaptation/actions/runs/45807162902
- Artifact ID: `15109497692`
- Artifact SHA-256: `ad23404cd44e5e2036bc250be00cca3044f314ebf8cdac2891ec5eef8993d411`
- Sample: 12 named objects, 132 directly paired gloved-hand participant-object recordings.
- Modality read: pressure arrays, master timestamps, tactile source timestamps, permitted object/participant metadata. No RGB/video arrays decoded.

## 1. Critical timestamp correction

The pressure sequence is aligned to `data.timestamps`, the synchronized master-clock timeline (approximately 15 Hz). `data.tactile_timestamps` is the glove source-update clock; it can repeat when the most recent glove readout is held at a later master tick.

Therefore:
- use `data.timestamps` as the pressure signal's time axis;
- retain every pressure row on the master clock;
- do not collapse rows merely because `tactile_timestamps` repeats;
- keep `tactile_timestamps` as an auxiliary source-clock diagnostic;
- process each participant-object episode independently.

In the 12-object sample, all 132 master-clock sequences were strictly increasing and finite. Median master interval was approximately 0.06745 s; median recording duration was 28.0 s (10th–90th percentile: 22.2–34.1 s).

The earlier Stage 0 count of 688 was based on incorrectly using the source tactile clock as the primary timeline. That count is superseded. The correct Stage 0 data-count pool is all **692 directly paired gloved-hand recordings**; the three apparently short source-clock records have normal-length master-clock sequences. A duplicate source-clock/pressure mismatch is retained as a warning, not used to exclude a master-clock-valid recording.

## 2. Recording and repetition unit

One OXT episode is one participant-object recording. The paper's acquisition protocol describes five repeated grasp attempts per participant and object, approximately six seconds each. The 1/2/5-shot Support unit is therefore a distinct participant-object recording, not an individual grasp attempt. All repeated pressure events from one participant remain nested within that participant's recording.

The 12-object sample covers all seven paper-level categories after grouping finer raw metadata labels (Tools + Electronics; Sports/Exercise within Sports & Toys).

## 3. Pressure-rise and repeated-event diagnostics

A simple sustained per-sensor 5 N threshold is already active at the beginning of all 132 recordings. Its first threshold crossing is therefore at recording time zero and cannot be interpreted as the physical contact onset. We do not use it as the event boundary.

Instead, an exploratory pressure-only event candidate was evaluated:
1. reconstruct 137 physical sensors from the 162 layout channels;
2. aggregate them into thumb, index, middle, ring, little, and palm regional loads;
3. bin total hand load into 0.5-second intervals;
4. smooth the binned series with a three-bin moving average;
5. detect peaks with minimum separation 3 seconds and prominence equal to 25% of the 95th–10th percentile dynamic range;
6. define a pressure-rise onset as the first three consecutive master-clock samples crossing 20% of the local trough-to-peak rise.

This is an operational pressure event definition, **not a ground-truth physical contact label**. The parameters are exploratory and must be checked on all 63 objects before freezing.

## 4. Cycle-count sensitivity

| Prominence fraction | Recordings with 5 peaks | 6 peaks | 4 peaks | 3 peaks | 0 peaks |
|---:|---:|---:|---:|---:|---:|
| 0.15 | 112 | 19 | 0 | 0 | 1 |
| 0.20 | 113 | 18 | 0 | 0 | 1 |
| 0.25 | **114** | 17 | 0 | 0 | 1 |
| 0.30 | 113 | 17 | 1 | 0 | 1 |

At the 0.25 setting:
- 114/132 recordings (86.4%) have exactly five pressure peaks;
- 17 recordings have six candidate peaks;
- one recording (GlassBowl / p001) has no detected peak;
- 672 candidate pressure events are produced in total.

Do not force every recording to have exactly five events. The 6-peak cases remain flagged for sensitivity analysis; the zero-peak case remains in the recording inventory but has no usable event sample under this detector.

## 5. Early/future window feasibility

For each candidate pressure event:
- onset = local pressure-rise onset;
- early input = [onset, onset + 1.0 s);
- future target = [onset + 1.0 s, onset + 3.0 s).

At the 0.25 prominence setting:
- all 672 detected events have non-overlapping early/future windows fully covered before the next detected event or recording end;
- all 672 future windows produce a finite six-region load-share vector;
- every six-region share vector sums to 1 (within floating-point precision).

Primary future target is the integrated regional load share:
`share_r = integral(future regional load_r) / integral(future total hand load)`.
This is a ratio of time-integrated loads, not an unweighted average of per-frame ratios.

## 6. Per-object cycle counts (prominence 0.25)

| Object | Recordings with exactly 5 detected peaks / 11 |
|---|---:|
| Apple | 9 |
| CannedLuncheonMeat | 9 |
| BodyWash | 8 |
| SprayBottle | 9 |
| Clip | 10 |
| TapeMeasure | 11 |
| Dumbbell | 8 |
| TennisBall | 10 |
| ElectricToothbrush | 11 |
| PowerBank-20000mAh | 10 |
| GlassBowl | 9 |
| Mug | 10 |

## 7. Stage 1 decision

**Conditional GO to full 63-object cycle/target inventory.**

Evidence:
- synchronized master time is valid across all 132 sample recordings;
- pressure-only event peaks are detectable in almost all sample recordings;
- the five-peak count is stable over prominence fractions 0.15–0.30;
- early/future windows and six-region future targets are valid for all 672 detected sample events.

Before the no-model Support test:
1. run the same frozen candidate detector on all 63 object archives;
2. quantify per-object and per-participant cycle-count variation;
3. inspect the 6-peak and zero-peak cases;
4. preserve all event counts and exclusions;
5. aggregate cycle-level data within participant before any Support/Query comparison.

No model was trained and no performance result was evaluated in this stage.
