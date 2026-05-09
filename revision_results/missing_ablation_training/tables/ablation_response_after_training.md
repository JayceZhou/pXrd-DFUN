# Ablation Response After Training

## Completion Status

The four previously missing configurations were trained with validation-based checkpoint selection and deterministic final inference. The completed table is available as `final_ablation_table_completed.csv` and `final_ablation_table_completed.tex`.

## Interpretation Guidance

Use the completed table to discuss the incremental contribution of peak descriptors, gate fusion, and FLS. Keep the split/protocol caveat visible: no-augmentation rows use the raw 239 split, while augmentation-based rows use the legacy augmented 245 split unless otherwise specified.

## Reviewer 2 Reply Suggestion

We trained the missing ablation configurations using fixed train/validation/test splits. Best checkpoints were selected only on validation weighted F1, and the held-out evaluation split was used once for final reporting. This avoids test-set checkpoint selection and provides per-sample prediction files for all newly trained rows.

## Conservative Language

Avoid claiming that every row is strictly comparable if the evaluation split differs. Emphasize protocol columns and report the completed table either in the main text or SI depending on space.
