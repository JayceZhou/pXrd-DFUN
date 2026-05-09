# Ablation Response Recommendation

## Completed Real Rows
The current real, traceable rows are CNN only, CNN + Augmentation, CNN + Augmentation + Peaks (Concat), CNN + Augmentation + Peaks (Gate), and CNN + Augmentation + Peaks (Gate) + FLS.

## Missing Reviewer-Requested Rows
The following rows still need real training before numerical values can be reported: Peaks only / MLP only; CNN + Peaks, no augmentation, concat; CNN + Peaks (Gate), no augmentation; CNN + Augmentation + FLS, no peak branch.

## Suggested Placement
Main manuscript: include the five confirmed rows with `Evaluation Split` and `Evidence Level`, or move split/protocol details to the caption if space is tight. SI: include the checkpoint audit, missing configurations, and training-needed plan.

## Conservative Reviewer 2 Reply Wording
The revised analysis adds a traceable ablation table for available checkpoints and explicitly separates rows that require new training. Existing results show stepwise gains from augmentation, peak descriptors, gated fusion, and the final Gate+FLS model. We do not assign numerical performance to configurations for which no checkpoint, prediction file, or training log exists.

## Protocol Caveat
State that the CNN-only ablation was recovered under the raw 239 split, whereas the augmentation and final model rows use the legacy augmented 245 split. Do not claim all rows were trained and selected under an identical protocol unless the missing rows are retrained under a unified script.
