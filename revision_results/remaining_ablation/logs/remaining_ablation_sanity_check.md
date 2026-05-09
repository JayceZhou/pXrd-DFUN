# Remaining Ablation Sanity Check

- Confirmed configurations: `cnn_only`, `cnn_aug`, `cnn_aug_peaks_concat`, `cnn_aug_peaks_gate`, `cnn_aug_peaks_gate_fls_final`.
- Unverified candidates: `exp_test1_model.pth`, `exp_test3_model.pth`, and `topk_model.pth` are 12224-dimensional gate-capable checkpoints, but their no-augmentation/FLS/concat roles are unverified.
- Incompatible candidate: `exp_test2_model.pth` has classifier input dim 12192 and does not match the current 12160/12224 adapter.
- Old variants: `PhyNetCNN_best_model.pth` and `best_model.pth` are 112-dimensional CNN+MLP variants, not peaks-only MLP rows.
- Missing checkpoints: Peaks only / MLP only; CNN + Peaks no-augmentation concat; CNN + Peaks no-augmentation gate; CNN + Augmentation + FLS no peak branch.
- Existing corrected mapping results were reused from `revision_outputs/predictions/corrected_ablation_mapping/`.
- Final Gate+FLS clean results were reused from `revision_outputs/gate_perturbation/` clean seed 0.
- No new deterministic inference was needed because no new confirmed/likely target checkpoint for A-D was found.
- No training was started.
- FLS implementation exists, but final checkpoint FLS status remains user-confirmed rather than independently run-config verified.
- Split caveat: `cnn_only` uses raw 239 split; all other included rows use legacy augmented 245 split.
- Rows needing retraining must not enter the manuscript with fabricated or placeholder values.
- Historical logs use `test_acc` naming; keep this as a conservative protocol caveat.
