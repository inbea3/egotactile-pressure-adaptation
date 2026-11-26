# Public Dataset Audit Protocol

## Objective

Determine whether a candidate public tactile dataset can support pressure-only few-shot adaptation to held-out objects or contact structures.

## Required audit record

For each candidate, create a record containing:

| Field | Required evidence |
|---|---|
| Dataset name/version | Official documentation or release |
| Source and access date | Stable URL and retrieval date |
| License | Exact license text or official statement |
| Modality | Pressure-only availability; identify other modalities separately |
| Sensor | Hardware, channel layout, resolution, sampling rate |
| Labels | Exact object/contact/task labels and definitions |
| Group identifiers | Subject, object, session, trial, sequence identifiers |
| Scale | Counts at raw-recording and independent-group levels |
| Missingness | Missing channels, corrupted records, incomplete labels |
| Splits | Proposed group-wise train/validation/test separation |
| Leakage risks | Repeated trials, temporal overlap, duplicate recordings |
| Suitability | Pass / conditional / fail, with reasons |

## Audit procedure

1. Read the official dataset paper, documentation, and license.
2. Inspect the downloadable metadata and directory structure before bulk download.
3. Confirm that pressure measurements can be extracted independently of video.
4. Map raw fields to a documented schema; retain original labels.
5. Count independent objects, structures, subjects, sessions, and trials.
6. Inspect representative records and verify units, shape, timestamps, and missingness.
7. Propose group-wise splits and test for overlap.
8. Record unresolved questions and decide pass / conditional / fail.

## Acceptance criteria

A primary dataset must have a legal and reproducible access path, usable pressure signals, sufficiently reliable labels for the stated task, and enough independent groups to form held-out evaluation tasks.

A large number of windows does not compensate for too few independent objects or sessions.

## Data handling

Do not commit raw datasets, credentials, private tokens, or restricted files. Store acquisition instructions and checksums where permitted. Keep derived data versioned by script and configuration.
