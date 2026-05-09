# Remaining Ablation Checkpoint Audit

## Scope
This audit checks whether the reviewer-requested missing ablations have real, traceable checkpoints or prediction files. No checkpoint weights were modified and no new training was run.

## Target Configurations
| Standard name | Target configuration | Evidence level | Checkpoint status | Inference status | Decision |
| --- | --- | --- | --- | --- | --- |
| `peaks_only_mlp` | Peaks only / MLP branch only | unverified/missing | no matching checkpoint found | not run | requires training |
| `cnn_peaks_concat_no_aug` | CNN + Peaks, no augmentation, concat | unverified/missing | no confirmed checkpoint found | not run | requires training |
| `cnn_peaks_gate_no_aug` | CNN + Peaks (Gate), no augmentation | unverified/missing | no confirmed checkpoint found | not run | requires training |
| `cnn_aug_fls_no_peak` | CNN + Augmentation + FLS, no peak branch | unverified/missing | no confirmed checkpoint found | not run | requires training |
| `cnn_aug_peaks_gate_fls_final` | CNN + Augmentation + Peaks (Gate) + FLS | confirmed with user evidence | `training_results/exp_results/exp_test4_model.pth` | existing clean inference reused | included in final table |

## Candidate Checkpoints Reviewed
| Checkpoint | Structure evidence | Candidate interpretation | Evidence for target config | Load/inference status | Evidence level | Table suitability |
| --- | --- | --- | --- | --- | --- | --- |
| `training_results/ablation/ablation1_model.pth` | classifier input dim 12160; output 230 | CNN only / xrd_only | user-corrected mapping confirms CNN only; raw 239 split | already inferred | confirmed | include, with split note |
| `training_results/ablation/ablation2_model.pth` | classifier input dim 12160; output 230 | CNN + Augmentation / xrd_only | user-corrected mapping confirms augmentation; no FLS evidence | already inferred | confirmed | include |
| `training_results/ablation/ablation3_model.pth` | classifier input dim 12224; mlp and gate keys present | CNN + Augmentation + Peaks (Concat) under corrected mapping | user-corrected mapping requires concat; no FLS | already inferred with forced concat | confirmed | include |
| `training_results/ablation/ablation4_model.pth` | classifier input dim 12224; mlp and gate keys present | CNN + Augmentation + Peaks (Gate), no FLS | user-corrected mapping confirms gate no-FLS | already inferred | confirmed | include |
| `training_results/exp_results/exp_test4_model.pth` | classifier input dim 12224; `cnn_branch`, `mlp_branch`, `gating_network`; output 230 | final Gate + FLS | user-confirmed complete model; no independent loss config found | existing clean inference reused | confirmed_with_user_evidence | include with evidence caveat |
| `training_results/exp_results/exp_test1_model.pth` | classifier input dim 12224; gate-capable | possible gate/concat model | no proof of no-augmentation, concat training, or FLS status | not used | unverified | exclude |
| `training_results/exp_results/exp_test2_model.pth` | classifier input dim 12192, incompatible with current 12160/12224 adapter | unknown historical variant | no proof of xrd_only FLS no-peak; dimension mismatch for current adapter | not used | incompatible/unverified | exclude |
| `training_results/exp_results/exp_test3_model.pth` | classifier input dim 12224; gate-capable, key count differs | possible gate model | no proof of no-augmentation or FLS/no-FLS role | not used | unverified | exclude |
| `training_results/npj_test/topk_model.pth` | classifier input dim 12224; gate-capable | current train.py gate model | current `train.py` uses CrossEntropyLoss with FLS line commented; not no-augmentation target | loadable but not target | likely gate CE, not target | exclude |
| `training_results/PhyNetCNN_best_model.pth` / `training_results/best_model.pth` | classifier input dim 112; old cnn+mlp, physical input 32, no gate | historical PhyNetCNN variant | not peaks-only; not current peak descriptors; no target protocol evidence | not used | incompatible/unverified | exclude |
| baseline/rruff/sim patch/cnn/reg checkpoints | non-PhyNetCNN key structures or other datasets | baselines | not target ablation configurations | not used | incompatible | exclude |

## FLS Evidence Boundary
`models/modules/FocalLoss_LabelSmoothing.py` implements FLS and `train.py` imports it, but the current training entry has `criterion = FocalLossWithLabelSmoothing()` commented out. The final checkpoint is therefore marked as user-confirmed complete-model FLS, not independently log-verified.

## Test-Selection / Protocol Risk
Historical JSON files often use `test_acc` naming and current `train.py` saves by validation loss only in the visible entrypoint. Earlier ablation code may have selected on test-like metrics. The final table therefore preserves `Evaluation Split` and `Evidence Level` columns and should not overstate strict protocol equivalence.
