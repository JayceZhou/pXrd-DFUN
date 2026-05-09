# FLS Minority-Class Sanity Check

## Inputs

- Final Gate + FLS prediction file: `revision_outputs/gate_perturbation/per_sample/final_gate_fls_model_gate_perturbation_per_sample.csv`
- Gate no-FLS prediction file: `revision_outputs/predictions/corrected_ablation_mapping/ablation4_cnn_aug_peaks_gate_opxrd/per_sample_predictions.csv`
- Both files were filtered to clean deterministic predictions. For the FLS gate-perturbation file, `seed == 0` was used.

## Alignment

- Same evaluation split: yes, validated by exact `sample_id` set equality.
- Number of aligned samples: 245
- `sample_id` aligned: True
- True labels aligned: True
- Number of evaluated classes: 56
- Classes without test samples among 230 labels: 174

## Class Counts

- Class-count rows missing after merge: 0
- Space-group/count mapping mismatches: 0
- Frequency groups present in this split: Head, Medium, Tail
- Grouping rule: Tail = lowest 30% of training counts, Medium = middle 40%, Head = highest 30%, from `revision_outputs/metrics/class_counts.csv`.
- Tail classes with fewer than 3 test samples: 8

## Model Evidence Boundary

- `final_gate_fls_model` uses `training_results/exp_results/exp_test4_model.pth`. It is treated as Gate + FLS based on user confirmation of the complete-model experiment; the repository does not contain an independent saved run command/config proving the exact loss name for this checkpoint.
- `gate_no_fls_model` is represented by corrected mapping `ablation4_cnn_aug_peaks_gate` / `training_results/ablation/ablation4_model.pth`, documented as CNN + Augmentation + Peaks (Gate) without FLS.
- No checkpoint selection was performed in this analysis.

## Overall Metrics

- No-FLS: Accuracy 0.693878, Weighted F1 0.681020, Macro F1 0.452811, Top-3 0.751020
- FLS: Accuracy 0.726531, Weighted F1 0.717924, Macro F1 0.511299, Top-3 0.763265

## Suitability

- Main text candidate: grouped Head/Medium/Tail Recall/Macro F1 figure and compact manuscript table.
- SI candidate: full class-wise metrics and Tail-class improvement table.
