# Gate Perturbation Sanity Check

- final_gate_fls_model checkpoint confirmed: True
- checkpoint_path: `training_results/exp_results/exp_test4_model.pth`
- checkpoint_load_success: yes
- fusion_mode: gate
- FLS status: verified
- FLS evidence: user-confirmed complete-model experiment checkpoint; no saved
  run config or training log in the repository independently records the loss
  function for this checkpoint.
- split_indices_path: `revision_outputs/configs/final_gate_fls_eval_split_indices.json`
- split_source: legacy augmented 245 split; selected for final_gate_fls_model because complete-model experiment checkpoint is associated with the legacy augmented opXRD protocol
- num_eval_samples: 245
- gate definition: PhyNetCNN.Model.forward computes g = gating_network(cat(xrd_features, phys_features)); gated_xrd_features = g * xrd_features and gated_phys_features = (1 - g) * phys_features. Larger gate_weight means stronger scaling of the CNN/XRD branch and weaker scaling of the peak/physical-feature branch. It is a scalar feature gate, not a calibrated branch probability.
- sample counts by perturbation/seed: {('clean', 0): 245, ('clean', 1): 245, ('clean', 2): 245, ('clean', 3): 245, ('clean', 4): 245, ('peak_elimination', 0): 245, ('peak_elimination', 1): 245, ('peak_elimination', 2): 245, ('peak_elimination', 3): 245, ('peak_elimination', 4): 245, ('preferred_orientation_like', 0): 245, ('preferred_orientation_like', 1): 245, ('preferred_orientation_like', 2): 245, ('preferred_orientation_like', 3): 245, ('preferred_orientation_like', 4): 245, ('strong_peak_scaling', 0): 245, ('strong_peak_scaling', 1): 245, ('strong_peak_scaling', 2): 245, ('strong_peak_scaling', 3): 245, ('strong_peak_scaling', 4): 245}
- gate extracted: yes
- gate values all in [0, 1]: True
- probabilities row-sum range: 1.000000 to 1.000000
- entropy range: 0.040674 to 5.305376; theoretical max log(230) = 5.438079
- top-3 calculated: True
- strong_peak_scaling peak descriptors: recomputed from perturbed spectrum
- peak_elimination peak descriptors: recomputed from perturbed spectrum
- preferred_orientation_like: approximate peak-intensity reweighting, not hkl/March-Dollase
- clean accuracy: 0.726531, matching the `training_results/exp_results/exp_test4_acc.json`
  max accuracy scale for the selected complete-model checkpoint.
- NaN values present: False
- max predicted-class fraction across perturbation/seed: 0.298
- misused ablation1/2/3 for main gate analysis: False
- no-FLS gate control saved separately: not determined by this script

Clean summary:

model_name,fls_status,perturbation_type,num_seeds,accuracy_mean,accuracy_std,weighted_f1_mean,weighted_f1_std,macro_f1_mean,macro_f1_std,top3_acc_mean,top3_acc_std,entropy_mean,entropy_std,gate_mean,gate_median,gate_std_across_samples,gate_std_across_seeds,delta_gate_vs_clean,delta_accuracy_vs_clean,delta_weighted_f1_vs_clean,delta_macro_f1_vs_clean,delta_entropy_vs_clean
final_gate_fls_model,verified,clean,5,0.726530612244898,0.0,0.7179237646088538,0.0,0.5112993876625684,0.0,0.763265306122449,0.0,1.8946176030809077,2.220446049250313e-16,0.1945421548047084,0.17747604846954346,0.09076335772476722,0.0,0.0,0.0,0.0,0.0,0.0
final_gate_fls_model,verified,strong_peak_scaling,5,0.6759183673469387,0.010829795233813562,0.663381335931731,0.00955647495393825,0.4556655423069772,0.01056347257862702,0.7559183673469387,0.007481756236662608,2.2944120231583214,0.022598808450857908,0.2059901354336465,0.19672071933746338,0.0864526650861859,0.0027153141902277697,0.011447980628938093,-0.050612244897959346,-0.05454242867712278,-0.05563384535559124,0.3997944200774137
final_gate_fls_model,verified,peak_elimination,5,0.6171428571428571,0.012751428042296589,0.6208512461126333,0.01084744225726046,0.43309713936958305,0.018042817249111085,0.7273469387755102,0.006530612244897959,2.479166566468745,0.04469832745066034,0.21163551368579575,0.21476754546165466,0.09017820159226977,0.00291654904203892,0.017093358881087345,-0.1093877551020409,-0.09707251849622045,-0.07820224829298539,0.5845489633878374
final_gate_fls_model,verified,preferred_orientation_like,5,0.6261224489795918,0.019421024087135245,0.6191971769162912,0.017206811161253385,0.3969549881031706,0.028407908649506527,0.7412244897959184,0.01113320954855986,2.471126835538428,0.048007294253448934,0.21128885262170613,0.20518170297145844,0.08472161966527192,0.0020424038978285423,0.016746697816997724,-0.10040816326530622,-0.0987265876925626,-0.11434439955939785,0.5765092324575205
