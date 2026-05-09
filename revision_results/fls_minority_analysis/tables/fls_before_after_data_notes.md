# FLS Before/After Minority Data Notes

## Models Compared

- Before / no-FLS: CNN + Augmentation + Peaks (Gate) = ablation4_cnn_aug_peaks_gate.
- After / FLS: CNN + Augmentation + Peaks (Gate) + FLS = final_gate_fls_model.
- Both are aligned on the same legacy augmented 245-sample evaluation split.

## Minority / Underrepresented Definitions

Primary grouping is based on training-set class frequency from revision_outputs/metrics/class_counts.csv:

- Tail: lowest 30% of classes by training sample count.
- Medium: middle 40%.
- Head: highest 30%.

Additional support-threshold summaries are provided for n_test_samples <= 3 and n_test_samples <= 5.

## Output Files

- revision_outputs/fls_minority_analysis/tables/fls_classwise_before_after_requested_format.csv
- revision_outputs/fls_minority_analysis/tables/fls_minority_group_summary_requested_format.csv
- revision_outputs/fls_minority_analysis/per_sample/fls_vs_no_fls_sample_predictions_requested_format.csv
