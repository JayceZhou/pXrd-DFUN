# Final Gate + FLS Checkpoint Verification

## Decision

Selected `final_gate_fls_model` checkpoint:

- `training_results/exp_results/exp_test4_model.pth`

This checkpoint is used because the user identified the complete-model
experiment checkpoint for the final model, and this artifact matches the
complete-model metric scale in the existing experiment logs.

Important caveat: the repository does not contain a saved run config or training
command that independently records the loss function for this checkpoint.
Therefore the FLS status is treated as user-confirmed / experiment-assigned, not
log-verified. The analysis below does not use the perturbation results for
checkpoint selection.

## Gate Definition

The current `models/PhyNetCNN.py` gate formula is:

```python
gated_xrd_features = g * xrd_features
gated_phys_features = (1.0 - g) * phys_features
```

Thus, larger `gate_weight` indicates stronger reliance on the CNN/XRD branch;
smaller `gate_weight` indicates stronger reliance on the peak/physical-feature
branch.

## Candidate Checkpoints

| Checkpoint | Fusion / structure evidence | Peak branch | Gate | Output classes | FLS evidence | Augmentation evidence | Load status | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `training_results/exp_results/exp_test4_model.pth` | `classifier_head.0.weight` input dim 12224; `cnn_branch`, `mlp_branch`, and `gating_network` keys present | yes | yes | 230 | user-confirmed complete-model experiment; no saved config/log names the loss | legacy augmented protocol inferred from experiment logs | loadable with `fusion_mode=gate` | selected as `final_gate_fls_model` |
| `training_results/ablation/ablation4_model.pth` | input dim 12224; gated PhyNetCNN keys present | yes | yes | 230 | no FLS evidence; corrected mapping says this is Gate without FLS | legacy augmented protocol | loadable with `fusion_mode=gate` | suitable as optional `gate_no_fls_ablation` SI control |
| `training_results/npj_test/topk_model.pth` | input dim 12224; gated PhyNetCNN keys present | yes | yes | 230 | current `train.py` uses CrossEntropyLoss; FLS line is commented | current train.py applies train/val/test split | loadable with `fusion_mode=gate` | not selected as final Gate+FLS |
| `training_results/exp_results/exp_test1_model.pth` | input dim 12224; gated PhyNetCNN keys present | yes | yes | 230 | no FLS evidence | unknown | likely loadable | not selected |
| `training_results/exp_results/exp_test3_model.pth` | input dim 12224; gated PhyNetCNN keys present | yes | yes | 230 | no FLS evidence | unknown | architecture differs in key count but gate keys present | not selected |
| `training_results/ablation/ablation3_model.pth` | input dim 12224; corrected mapping requires concat inference | yes | no for logits under corrected mapping | 230 | no FLS evidence | legacy augmented protocol | loadable with `fusion_mode=concat` | not a gate model in corrected ablation mapping |
| `training_results/ablation/ablation1_model.pth`, `training_results/ablation/ablation2_model.pth` | input dim 12160; xrd-only classifier path | no peak branch in logits | no for logits | 230 | no FLS evidence | split differs by ablation | loadable with `fusion_mode=xrd_only` | not eligible for gate analysis |

## Evidence Summary

- Fusion mode: `training_results/exp_results/exp_test4_model.pth` has a
  12224-dimensional classifier input, consistent with concatenated CNN and
  physical peak features.
- Peak / physical branch: `mlp_branch` state-dict keys are present.
- Gate branch: `gating_network` state-dict keys are present.
- Output classes: final classifier tensors are compatible with 230 classes.
- FLS: `models/modules/FocalLoss_LabelSmoothing.py` implements focal loss plus
  label smoothing. However, no saved config or log ties this loss to
  `exp_test4_model.pth`. The current `train.py` has
  `criterion = FocalLossWithLabelSmoothing()` commented out. `finetune.py` uses
  FLS but trains `NewNet`, not the gated PhyNetCNN checkpoint.
- Augmentation: the full-model experiment result is associated with the legacy
  augmented opXRD protocol based on the user-provided experiment mapping and the
  existing experiment metric scale.
- Test-selection risk: historical logs store `test_acc` and some training code
  has saved checkpoints based on validation or test-like metrics. The current
  perturbation analysis must not be used to select a checkpoint.

## Final Labels For This Round

- `final_gate_fls_model`: `training_results/exp_results/exp_test4_model.pth`
- `gate_no_fls_ablation`: `training_results/ablation/ablation4_model.pth`

The main perturbation analysis is run only for `final_gate_fls_model` in this
round. The no-FLS gate checkpoint remains available for SI comparison later.
