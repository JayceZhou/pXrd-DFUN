# Training Capability Audit

## Current Training Entrypoint

- Existing script: `train.py`.
- It does not expose CLI switches for `use_xrd_branch`, `use_peak_branch`, `fusion_mode`, `use_augmentation`, or `use_fls`.
- It imports `FocalLossWithLabelSmoothing`, but the visible active criterion is `CrossEntropyLoss`; the FLS criterion line is commented.
- It uses `PhyNetCNN.Model()` for the current visible entrypoint and historically saved checkpoints under `training_results/`.

## Added Revision Training Script

- Added `scripts/revision/train_missing_ablation.py` as a self-contained revision training entrypoint.
- This avoids modifying existing checkpoint weights or the original `train.py` training behavior.

## Supported Configurations In The Revision Script

1. `peaks_only_mlp`: supported via a new minimal `PeakOnlyMLP` using only 45-dimensional peak/physical descriptors from `HybridFeatureDataset.extract_physical_features`.
2. `cnn_peaks_concat_no_aug`: supported via `PhyNetCNNRevisionAdapter` with `fusion_mode=concat`, raw no-augmentation training split.
3. `cnn_peaks_gate_no_aug`: supported via `PhyNetCNNRevisionAdapter` with `fusion_mode=gate`, raw no-augmentation training split and gate output.
4. `cnn_aug_fls_no_peak`: supported via `PhyNetCNNRevisionAdapter` with `fusion_mode=xrd_only`, legacy one-shot augmented training protocol, and `FocalLossWithLabelSmoothing`.

## New CLI Parameters

`--models`, `--device`, `--epochs`, `--batch-size`, `--learning-rate`, `--patience`, `--seed`, `--num-workers`, `--val-fraction`, and `--output-dir`.

## Checkpoint Selection

All best checkpoints are selected only by validation weighted F1 on the validation split. The fixed test/evaluation split is used once after the best checkpoint is loaded.


## Commands Used In This Run

All four configurations were trained in one reproducible command:

```bash
conda run -n opxrd python scripts/revision/train_missing_ablation.py --device cuda:0 --epochs 300 --batch-size 64 --learning-rate 5e-4 --patience 25 --seed 42 --num-workers 0 --output-dir revision_outputs/missing_ablation_training
```

Per-configuration YAML files are:

- `revision_outputs/missing_ablation_training/configs/peaks_only_mlp.yaml`
- `revision_outputs/missing_ablation_training/configs/cnn_peaks_concat_no_aug.yaml`
- `revision_outputs/missing_ablation_training/configs/cnn_peaks_gate_no_aug.yaml`
- `revision_outputs/missing_ablation_training/configs/cnn_aug_fls_no_peak.yaml`

Each model subdirectory also contains `training_command.txt`, `config.yaml`, `training_history.csv`, `training_curves.png`, `best_checkpoint.pth`, and `final_checkpoint.pth`.
