# Training-Needed Plan For Missing Ablations

No long training was started in this round. The following rows require real training before numerical results can be entered.

## `peaks_only_mlp`
- Target: peak / physical descriptor-only classifier.
- Modules: disable CNN/XRD branch; use only `HybridFeatureDataset` physical descriptors (`x_phys`).
- Fusion mode: none / physical-only.
- Augmentation: record explicitly; use the selected comparison protocol.
- FLS: off unless a separate FLS-specific row is desired.
- Training script status: current `train.py` does not expose a CLI or physical-only model path. Add a small `PhysicalOnlyMLP` model or a `fusion_mode=phys_only` adapter.
- Command template after code support exists: `python train.py <real project args for physical-only MLP>`.
- Expected checkpoint: `training_results/remaining_ablation/peaks_only_mlp_model.pth`.
- Integration: run a physical-only inference adapter and append to `remaining_ablation_metrics.csv`.

## `cnn_peaks_concat_no_aug`
- Target: CNN + peak descriptors, concat fusion, no training augmentation.
- Modules: CNN branch enabled; physical branch enabled; gate disabled in logits.
- Fusion mode: concat.
- Augmentation: off for training.
- FLS: off.
- Training script status: current `train.py` does not expose a CLI flag for concat/gate/no-augmentation. The inference adapter supports concat, but training code must explicitly use concat logits.
- Command template after code support exists: `python train.py <real project args: fusion_mode=concat, augmentation=off, fls=off>`.
- Expected checkpoint: `training_results/remaining_ablation/cnn_peaks_concat_no_aug_model.pth`.
- Integration: run deterministic inference with `--fusion-mode concat`.

## `cnn_peaks_gate_no_aug`
- Target: CNN + peak descriptors, gate fusion, no training augmentation.
- Modules: CNN branch enabled; physical branch enabled; gate enabled.
- Fusion mode: gate.
- Augmentation: off for training.
- FLS: off.
- Training script status: current `train.py` gate model can be adapted, but no CLI config currently records no-augmentation gate training.
- Command template after code support exists: `python train.py <real project args: fusion_mode=gate, augmentation=off, fls=off>`.
- Expected checkpoint: `training_results/remaining_ablation/cnn_peaks_gate_no_aug_model.pth`.
- Integration: run deterministic inference with `--fusion-mode gate` and save gate weights.

## `cnn_aug_fls_no_peak`
- Target: CNN + Augmentation + FLS, no peak branch.
- Modules: CNN/XRD branch enabled; physical branch disabled from logits.
- Fusion mode: xrd_only.
- Augmentation: on for training.
- FLS: on (`FocalLossWithLabelSmoothing`).
- Training script status: `train.py` imports FLS but the active criterion is `CrossEntropyLoss`; needs an explicit config or edit to select FLS and xrd_only logits.
- Command template after code support exists: `python train.py <real project args: fusion_mode=xrd_only, augmentation=on, fls=on>`.
- Expected checkpoint: `training_results/remaining_ablation/cnn_aug_fls_no_peak_model.pth`.
- Integration: run deterministic inference with `--fusion-mode xrd_only`.
