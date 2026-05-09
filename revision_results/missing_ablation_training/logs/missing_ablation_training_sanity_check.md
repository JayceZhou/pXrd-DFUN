# Missing Ablation Training Sanity Check

- Models completed: peaks_only_mlp, cnn_peaks_concat_no_aug, cnn_peaks_gate_no_aug, cnn_aug_fls_no_peak
- Best checkpoint selection split: validation for all models.
- Best checkpoint metric: validation weighted F1 for all models.
- Test-selected checkpoint risk: no; fixed test split evaluated once after loading best validation checkpoint.

## peaks_only_mlp
- checkpoint saved: True
- evaluation split: raw 239 split
- samples: 239
- output classes: 230
- probability sums min/max: 1.000000 / 1.000000
- NaN in probabilities: False
- training augmentation: no
- peak branch: enabled
- FLS: no_fls
- fusion mode: peak_only

## cnn_peaks_concat_no_aug
- checkpoint saved: True
- evaluation split: raw 239 split
- samples: 239
- output classes: 230
- probability sums min/max: 1.000000 / 1.000000
- NaN in probabilities: False
- training augmentation: no
- peak branch: enabled
- FLS: no_fls
- fusion mode: concat

## cnn_peaks_gate_no_aug
- checkpoint saved: True
- evaluation split: raw 239 split
- samples: 239
- output classes: 230
- probability sums min/max: 1.000000 / 1.000000
- NaN in probabilities: False
- training augmentation: no
- peak branch: enabled
- FLS: no_fls
- fusion mode: gate

## cnn_aug_fls_no_peak
- checkpoint saved: True
- evaluation split: legacy augmented 245 split
- samples: 245
- output classes: 230
- probability sums min/max: 1.000000 / 1.000000
- NaN in probabilities: False
- training augmentation: yes
- peak branch: disabled
- FLS: fls
- fusion mode: xrd_only

## Configuration Assertions
- `peaks_only_mlp` uses only computed peak/physical descriptors and no raw XRD branch.
- `cnn_peaks_concat_no_aug` and `cnn_peaks_gate_no_aug` use the raw no-augmentation split and no materialized one-shot augmentation.
- `cnn_aug_fls_no_peak` uses the legacy one-shot augmented training dataset, FocalLossWithLabelSmoothing, and xrd_only logits with the peak branch disabled from logits.
