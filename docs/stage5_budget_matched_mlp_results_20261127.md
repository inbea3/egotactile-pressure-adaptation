# Stage 5 follow-up: longer-trained MLP reverses the apparent MAML advantage

**Status:** exploratory, corrective analysis on the previously inspected 63 EgoTactile-OXT objects. This report supersedes any claim that Stage 5 established MAML superiority over supervised learning.

## Why this check was needed

The original MLP used 1,500 Adam steps with batch size 3, or 4,500 training-record exposures. Exact MAML used 1,500 object tasks, each with three inner Support steps and a distinct Query pass. With uniformly sampled 1/2/5-shot Support, this is approximately 16,500 record exposures. Equal outer iterations therefore gave MAML substantially more exposure to the training data.

We reran the same supervised MLP architecture with **5,500 Adam steps × batch size 3 ≈ 16,500 record exposures**. This is an approximate exposure match, not exact equality of floating-point operations, optimizer calls, or wall time. Test adaptation remained three SGD steps at rate 0.05. The five object folds, deterministic Support combinations, three training seeds, target, and evaluation pipeline remained unchanged. The Stage 4 pilot script was called with `--supervised-steps 5500 --meta-iterations 0`; its unused Reptile output is ignored.

For each method, we averaged each object's Query MAE over the three seeds and then bootstrapped whole objects 5,000 times. Positive MLP-minus-MAML differences favor MAML. These intervals do not incorporate method selection on the same 63 objects or full training-seed uncertainty.

## Results

| Shots | Original MLP, adapted (1,500 steps) | Longer MLP, unadapted | Longer MLP, adapted | Exact MAML, adapted |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.6866 | **0.6367** | 0.6614 | 0.6767 |
| 2 | 0.6721 | **0.6367** | 0.6465 | 0.6604 |
| 5 | 0.6493 | 0.6367 | **0.6247** | 0.6407 |

At 5 shots, longer-trained MLP fine tuning beats exact MAML by **0.01595 MAE**. The paired MLP-minus-MAML difference is `-0.01595`, object-bootstrap 95% CI `[-0.02280, -0.00907]`; exact MAML wins on only 15/63 objects. The longer MLP's 5-shot MAE is 0.62573, 0.62367, and 0.62469 across the three seeds, versus 0.63575, 0.64434, and 0.64185 for exact MAML. The direction is consistent across seeds.

At 1 and 2 shots, the **unadapted** longer MLP is the strongest method in this comparison, at MAE 0.6367. Fine tuning from one or two Support records worsens the longer MLP. This behavior cautions against reporting only adapted baselines.

## Consequences for the research claim

- Exact MAML still shows substantial improvement from its own unadapted initialization at 5 shots, but its absolute performance does not beat a longer-trained supervised MLP. The earlier apparent MAML advantage arose in a comparison with a weaker, shorter-trained MLP.
- There is currently **no evidence that MAML is the best method** for this coarse six-region pressure-change target. Further tuning on these same 63 inspected objects would compound selection bias. A new task, stronger mechanism hypothesis, or independent object set is needed before making a meta-learning superiority claim.
- These offline human-glove data do not yet measure sensor-to-sensor coordination edges or robot online adaptation. The raw EgoTactile-OXT sensor archives are publicly listed but are not downloaded in this AutoDL repository; the current Stage 1 report retains only six-region 0.5-second means.

## External-data audit

TouchScale exposes episode metadata (`episode_id`, `task_id`, free-text `items`, `actor_id`, and shard) and raw per-pad tactile arrays, so it is a plausible **candidate** for a new task-level audit. The `metadata/episodes.parquet` file is about 5.4 MB, but the Hugging Face repository is gated: an account must accept the dataset conditions before file access. This AutoDL container has no Hugging Face token or `huggingface_hub` installation, and a direct connection to `huggingface.co:443` timed out. Therefore participant-by-object counts and 5-shot eligibility **have not yet been verified**. The full data are about 1 TB and should not be downloaded before the metadata audit. Official dataset card: https://huggingface.co/datasets/2077AIDataFoundation/TouchScale .

EgoTouch HDF5 is an ungated alternative with normalized 21×21 pressure maps, but its public card does not establish participant IDs or an unambiguous repeated-object key. It cannot yet replace the TouchScale audit for the participant-disjoint new-object protocol. Card: https://huggingface.co/datasets/zhenyuxie-zhzh/EgoTouch_hdf5 .

Numeric result: `results/stage5_mlp_budget5500_summary.json`. Full per-seed/per-object outputs and logs remain on AutoDL as `results/stage5_mlp_budget5500_seed*.json` and `.log`. The original Stage 5 report is marked superseded at its top.

