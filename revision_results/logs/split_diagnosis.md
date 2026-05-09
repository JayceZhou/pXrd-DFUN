# Split Diagnosis

The unusually high Round 2 clean opXRD accuracies were caused by using a
different `random_state=42` split than the historical ablation checkpoints used.
The seed was the same, but the split protocol was not.

## Split Protocols Compared

### Round 2 Raw Holdout

- File: `revision_outputs/configs/opxrd_eval_split_indices.json`
- Source: raw `experiment_xrd.npz`
- Protocol: `train_test_split(indices, test_size=0.2, random_state=42, stratify=None)`
- Eval samples: 239 raw samples

This split reproduces the `candidate_ablation1` logged best accuracy exactly:
`0.635983 = 152 / 239`.

### Legacy Ablation Raw-Test Reconstruction

- File: `revision_outputs/configs/opxrd_legacy_ablation_test_raw_indices.json`
- Source: raw `experiment_xrd.npz` labels plus duplicated singleton-class labels
- Protocol reconstructed from, and later confirmed against, the older commented
  `get_experiment_dataloader` block in `train.py`:
  `augment_one_shot_classes` labels first, then
  `train_test_split(..., test_size=0.2, stratify=dataset.labels, random_state=42)`
- Historical augmented total: 1223 samples
- Historical test total: 245 samples
- Reproducible raw test samples: 244
- Excluded augmented test samples: 1
- Excluded augmented source index: 715

The excluded sample cannot be reproduced exactly because the generated
one-shot augmented pattern was not saved.

## Overlap Check

Comparing the Round 2 raw holdout against the reconstructed legacy ablation
split:

- Round 2 eval samples: 239
- Overlap with legacy raw test samples: 52
- Overlap with legacy raw train samples: 187

Therefore the Round 2 raw holdout should not be used to evaluate checkpoints
that were trained with the legacy ablation split, because most of that eval set
appears to be training data for those checkpoints.

## Metric Check

Metrics on the legacy raw-test reconstruction:

| model_name | Accuracy | Weighted F1 | Macro F1 | Top-3 Accuracy | n |
| --- | ---: | ---: | ---: | ---: | ---: |
| `candidate_ablation1` | 0.881148 | 0.861694 | 0.701850 | 0.942623 | 244 |
| `candidate_ablation2` | 0.659836 | 0.650033 | 0.418688 | 0.741803 | 244 |
| `candidate_ablation3` | 0.655738 | 0.656464 | 0.456736 | 0.725410 | 244 |
| `candidate_ablation4` | 0.692623 | 0.679713 | 0.443841 | 0.750000 | 244 |
| `candidate_exp_test4` | 0.725410 | 0.716768 | 0.503288 | 0.762295 | 244 |
| `candidate_npjt_topk` | 0.639344 | 0.613747 | 0.400485 | 0.725410 | 244 |

These values match the scale of the historical `ablation2`-`ablation4` logs
much better than the Round 2 raw-holdout metrics. For example:

- `ablation2_acc.json` max test accuracy: 0.657143; reconstructed raw-test:
  0.659836
- `ablation3_acc.json` max test accuracy: 0.665306; reconstructed raw-test:
  0.655738
- `ablation4_acc.json` max test accuracy: 0.689796; reconstructed raw-test:
  0.692623

`candidate_ablation1` behaves differently: its logged best accuracy matches the
Round 2 raw holdout, not the legacy ablation raw-test reconstruction. This is a
strong sign that ablation checkpoints were not all trained/evaluated with the
same split protocol.

## Full Augmented-Test Reproduction And Corrected Mapping

After the user confirmed that one historical ablation used the older two-way
`get_experiment_dataloader` block, the actual one-shot augmented samples were
materialized and inference was rerun on the full 245-sample test split.

The user later corrected the ablation mapping:

| Ablation | Configuration | Inference fusion |
| --- | --- | --- |
| 1 | CNN only | `xrd_only` |
| 2 | CNN + Augmentation | `xrd_only` |
| 3 | CNN + Augmentation + Peaks (Concat) | `concat` |
| 4 | CNN + Augmentation + Peaks (Gate) | `gate` |

Generated files:

- `scripts/revision/create_legacy_augmented_opxrd.py`
- `revision_outputs/tables/ablation_mapping.csv`
- `revision_outputs/configs/opxrd_legacy_augmented_seed42.npz`
- `revision_outputs/configs/opxrd_legacy_augmented_test_indices.json`
- `revision_outputs/predictions/legacy_ablation_augmented_test/`
- `revision_outputs/predictions/prediction_summary_legacy_ablation_augmented_test.csv`
- `revision_outputs/metrics/basic_metrics_by_model_legacy_ablation_augmented_test.csv`
- `revision_outputs/predictions/corrected_ablation_mapping/`
- `revision_outputs/predictions/prediction_summary_corrected_ablation_mapping.csv`
- `revision_outputs/metrics/basic_metrics_corrected_ablation_mapping.csv`

The recreated augmented dataset contains 1194 raw samples and 29 one-shot
augmented samples. The split contains 245 test samples, including one augmented
test item:

- augmented test index: 1218
- raw source index: 715

Metrics under the corrected mapping:

| model_name | Configuration | Accuracy | Weighted F1 | Macro F1 | Top-3 Accuracy | n |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `ablation1_cnn_only` | CNN only | 0.635983 | 0.623164 | 0.362560 | 0.719665 | 239 |
| `ablation2_cnn_augmentation` | CNN + Augmentation | 0.661224 | 0.651461 | 0.428541 | 0.742857 | 245 |
| `ablation3_cnn_aug_peaks_concat` | CNN + Augmentation + Peaks (Concat) | 0.669388 | 0.668002 | 0.474369 | 0.726531 | 245 |
| `ablation4_cnn_aug_peaks_gate` | CNN + Augmentation + Peaks (Gate) | 0.693878 | 0.681020 | 0.452811 | 0.751020 | 245 |

Historical log maxima:

| log | Max Accuracy | Max Weighted F1 |
| --- | ---: | ---: |
| `ablation1_acc.json` | 0.635983 | 0.626108 |
| `ablation2_acc.json` | 0.657143 | 0.654406 |
| `ablation3_acc.json` | 0.665306 | 0.663920 |
| `ablation4_acc.json` | 0.689796 | 0.676938 |

The full augmented-test reproduction confirms that ablations 2-4 are on the
expected historical scale when the corrected fusion modes are used. Ablation 1
is different: it matches the raw 239-sample split instead. Therefore ablation 1
should not be evaluated on the legacy augmented split unless old run records
prove it was trained with that same protocol.

The one-shot augmented spectrum is regenerated with fixed seed 42 for
reproducibility. The historical augmented spectrum was not saved, so exact
bitwise identity to the old run cannot be guaranteed.

## Outputs Generated

- Legacy raw-test split:
  `revision_outputs/configs/opxrd_legacy_ablation_test_raw_indices.json`
- Legacy inference directories:
  `revision_outputs/predictions/legacy_ablation_test/<model_name>_opxrd/`
- Legacy prediction summary:
  `revision_outputs/predictions/prediction_summary_legacy_ablation_raw_test.csv`
- Legacy metadata:
  `revision_outputs/predictions/prediction_summary_legacy_ablation_raw_test_metadata.json`
- Legacy metrics:
  `revision_outputs/metrics/basic_metrics_by_model_legacy_ablation_raw_test.csv`
- Corrected ablation mapping outputs:
  `revision_outputs/predictions/corrected_ablation_mapping/`
- Corrected ablation summary:
  `revision_outputs/predictions/prediction_summary_corrected_ablation_mapping.csv`
- Corrected ablation metrics:
  `revision_outputs/metrics/basic_metrics_corrected_ablation_mapping.csv`

## Recommendation

Do not use the Round 2 raw-holdout 90%+ metrics for `ablation2`-`ablation4` or
for paper claims. For reviewer-facing ablation analysis, use:

1. `prediction_summary_corrected_ablation_mapping.csv` for per-sample analysis.
2. `basic_metrics_corrected_ablation_mapping.csv` for aggregate metrics.
3. Explicit fusion-mode labels: ablation 1 and 2 are `xrd_only`, ablation 3 is
   `concat`, and ablation 4 is `gate`.

For the cleanest paper evidence, a fully consistent new train/validation/test
protocol and retraining of all ablations would still be stronger, but it is a
larger task.
