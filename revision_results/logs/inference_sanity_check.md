# Inference Sanity Check

- split_indices_path: `revision_outputs/configs/opxrd_eval_split_indices.json`
- split_source: deterministic raw opXRD holdout generated with sklearn.train_test_split; no official split file found
- augmentation_disabled: True

## candidate_ablation1_opxrd
- checkpoint_load_success: yes
- checkpoint_path: `training_results/ablation/ablation1_model.pth`
- model_structure_match: yes; strict revision adapter load completed for non-gated classifier variant `xrd_feature_classifier_12160_no_gate_in_logits`
- num_classes: 230
- num_samples: 239
- mc_dropout: False (single_softmax)
- gate_return_method: not_applicable_no_gate_in_logits
- opXRD eval split samples: 239 from 1194
- true label distribution: 53 classes; top counts {224: 73, 13: 23, 14: 13, 1: 11, 147: 10, 61: 9, 59: 8, 73: 7}
- pred label distribution: 35 classes; max class fraction 0.301; top counts {224: 72, 13: 35, 1: 25, 61: 11, 14: 9, 18: 9, 128: 7, 59: 6}
- probabilities saved: yes; shape [239, 230]
- logits saved: yes
- probability row sums: min 1.000000, max 1.000000
- entropy range: min 0.004036, mean 1.377398, max 4.081455; theoretical max log(230) = 5.438079
- top-3 calculated: yes
- gate weight extracted: no
- pred distribution abnormal concentration flag (>50% one class): no
- checkpoint/config mismatch found: no strict-load mismatch during this inference run
- test-selection risk: historical training code appears to have selected some checkpoints using test metrics; these inference outputs must not be used to choose the final checkpoint.

## candidate_ablation2_opxrd
- checkpoint_load_success: yes
- checkpoint_path: `training_results/ablation/ablation2_model.pth`
- model_structure_match: yes; strict revision adapter load completed for non-gated classifier variant `xrd_feature_classifier_12160_no_gate_in_logits`
- num_classes: 230
- num_samples: 239
- mc_dropout: False (single_softmax)
- gate_return_method: not_applicable_no_gate_in_logits
- opXRD eval split samples: 239 from 1194
- true label distribution: 53 classes; top counts {224: 73, 13: 23, 14: 13, 1: 11, 147: 10, 61: 9, 59: 8, 73: 7}
- pred label distribution: 51 classes; max class fraction 0.301; top counts {224: 72, 13: 22, 1: 13, 14: 12, 61: 10, 147: 9, 59: 7, 226: 6}
- probabilities saved: yes; shape [239, 230]
- logits saved: yes
- probability row sums: min 1.000000, max 1.000000
- entropy range: min 0.000002, mean 0.554186, max 4.152715; theoretical max log(230) = 5.438079
- top-3 calculated: yes
- gate weight extracted: no
- pred distribution abnormal concentration flag (>50% one class): no
- checkpoint/config mismatch found: no strict-load mismatch during this inference run
- test-selection risk: historical training code appears to have selected some checkpoints using test metrics; these inference outputs must not be used to choose the final checkpoint.

## candidate_npjt_topk_opxrd
- checkpoint_load_success: yes
- checkpoint_path: `training_results/npj_test/topk_model.pth`
- model_structure_match: yes; strict gated PhyNetCNN state-dict load completed for variant `unknown`
- num_classes: 230
- num_samples: 239
- mc_dropout: False (single_softmax)
- gate_return_method: forward_return_gate
- opXRD eval split samples: 239 from 1194
- true label distribution: 53 classes; top counts {224: 73, 13: 23, 14: 13, 1: 11, 147: 10, 61: 9, 59: 8, 73: 7}
- pred label distribution: 45 classes; max class fraction 0.301; top counts {224: 72, 13: 32, 61: 12, 14: 8, 147: 8, 18: 8, 1: 7, 73: 7}
- probabilities saved: yes; shape [239, 230]
- logits saved: yes
- probability row sums: min 1.000000, max 1.000000
- entropy range: min 0.000000, mean 1.066441, max 4.353088; theoretical max log(230) = 5.438079
- top-3 calculated: yes
- gate weight extracted: yes; min 0.001355, mean 0.440852, max 0.696716; all in [0, 1]: yes
- pred distribution abnormal concentration flag (>50% one class): no
- checkpoint/config mismatch found: no strict-load mismatch during this inference run
- test-selection risk: historical training code appears to have selected some checkpoints using test metrics; these inference outputs must not be used to choose the final checkpoint.

## candidate_exp_test4_opxrd
- checkpoint_load_success: yes
- checkpoint_path: `training_results/exp_results/exp_test4_model.pth`
- model_structure_match: yes; strict gated PhyNetCNN state-dict load completed for variant `unknown`
- num_classes: 230
- num_samples: 239
- mc_dropout: False (single_softmax)
- gate_return_method: forward_return_gate
- opXRD eval split samples: 239 from 1194
- true label distribution: 53 classes; top counts {224: 73, 13: 23, 14: 13, 1: 11, 147: 10, 61: 9, 59: 8, 73: 7}
- pred label distribution: 53 classes; max class fraction 0.305; top counts {224: 73, 13: 21, 1: 12, 61: 11, 14: 10, 147: 9, 59: 8, 73: 8}
- probabilities saved: yes; shape [239, 230]
- logits saved: yes
- probability row sums: min 1.000000, max 1.000000
- entropy range: min 0.031684, mean 1.045989, max 5.171630; theoretical max log(230) = 5.438079
- top-3 calculated: yes
- gate weight extracted: yes; min 0.010654, mean 0.185044, max 0.375156; all in [0, 1]: yes
- pred distribution abnormal concentration flag (>50% one class): no
- checkpoint/config mismatch found: no strict-load mismatch during this inference run
- test-selection risk: historical training code appears to have selected some checkpoints using test metrics; these inference outputs must not be used to choose the final checkpoint.

## candidate_ablation3_opxrd
- checkpoint_load_success: yes
- checkpoint_path: `training_results/ablation/ablation3_model.pth`
- model_structure_match: yes; strict gated PhyNetCNN state-dict load completed for variant `unknown`
- num_classes: 230
- num_samples: 239
- mc_dropout: False (single_softmax)
- gate_return_method: forward_return_gate
- opXRD eval split samples: 239 from 1194
- true label distribution: 53 classes; top counts {224: 73, 13: 23, 14: 13, 1: 11, 147: 10, 61: 9, 59: 8, 73: 7}
- pred label distribution: 54 classes; max class fraction 0.305; top counts {224: 73, 13: 18, 1: 12, 14: 11, 147: 9, 61: 8, 73: 8, 59: 8}
- probabilities saved: yes; shape [239, 230]
- logits saved: yes
- probability row sums: min 1.000000, max 1.000000
- entropy range: min 0.000216, mean 1.007167, max 4.738420; theoretical max log(230) = 5.438079
- top-3 calculated: yes
- gate weight extracted: yes; min 0.390900, mean 0.501210, max 0.704453; all in [0, 1]: yes
- pred distribution abnormal concentration flag (>50% one class): no
- checkpoint/config mismatch found: no strict-load mismatch during this inference run
- test-selection risk: historical training code appears to have selected some checkpoints using test metrics; these inference outputs must not be used to choose the final checkpoint.

## candidate_ablation4_opxrd
- checkpoint_load_success: yes
- checkpoint_path: `training_results/ablation/ablation4_model.pth`
- model_structure_match: yes; strict gated PhyNetCNN state-dict load completed for variant `unknown`
- num_classes: 230
- num_samples: 239
- mc_dropout: False (single_softmax)
- gate_return_method: forward_return_gate
- opXRD eval split samples: 239 from 1194
- true label distribution: 53 classes; top counts {224: 73, 13: 23, 14: 13, 1: 11, 147: 10, 61: 9, 59: 8, 73: 7}
- pred label distribution: 55 classes; max class fraction 0.305; top counts {224: 73, 13: 23, 14: 12, 1: 11, 61: 10, 147: 9, 59: 7, 18: 6}
- probabilities saved: yes; shape [239, 230]
- logits saved: yes
- probability row sums: min 1.000000, max 1.000000
- entropy range: min 0.000000, mean 0.109612, max 3.125615; theoretical max log(230) = 5.438079
- top-3 calculated: yes
- gate weight extracted: yes; min 0.242823, mean 0.882925, max 0.999964; all in [0, 1]: yes
- pred distribution abnormal concentration flag (>50% one class): no
- checkpoint/config mismatch found: no strict-load mismatch during this inference run
- test-selection risk: historical training code appears to have selected some checkpoints using test metrics; these inference outputs must not be used to choose the final checkpoint.
