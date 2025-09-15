from dataclasses import dataclass
import random
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report
from models import PhyNetCNN
from models import NoPoolCNN
from train import get_experiment_dataloader

def enable_dropout(model):
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def evaluate_with_uncertainty(model, data_loader, device, T=50):
    model.to(device)
    model.eval()
    enable_dropout(model)

    all_true_labels = []
    all_preds = []
    all_confidences = []
    all_uncertainties = []
    all_mean_probs = []
    
    with torch.no_grad():
        for inputs_dict, labels in tqdm(data_loader, desc="Evaluating with MC Dropout"):
            raw_xrd = inputs_dict['raw_xrd'].to(device)
            phys_feats = inputs_dict['physical_features'].to(device)

            batch_probs_T = []
            for _ in range(T):
                outputs = model(raw_xrd, phys_feats)
                # outputs = model(raw_xrd)
                probs = F.softmax(outputs, dim=1)
                batch_probs_T.append(probs.unsqueeze(0))

            all_probs_tensor = torch.cat(batch_probs_T, dim=0)
            mean_probs = all_probs_tensor.mean(dim=0)
            confidences, preds = torch.max(mean_probs, dim=1)

            uncertainties = -torch.sum(mean_probs * torch.log(mean_probs + 1e-9), dim=1)
            
            all_true_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_confidences.extend(confidences.cpu().numpy())
            all_uncertainties.extend(uncertainties.cpu().numpy())
            all_mean_probs.append(mean_probs.cpu().numpy())

    all_true_labels = np.array(all_true_labels)
    all_preds = np.array(all_preds)
    all_confidences = np.array(all_confidences)
    all_uncertainties = np.array(all_uncertainties)
    all_mean_probs = np.concatenate(all_mean_probs, axis=0)

    results_df = pd.DataFrame({
        'true_label': all_true_labels,
        'pred_label': all_preds,
        'confidence': all_confidences,
        'uncertainty': all_uncertainties
    })
    results_df['correct'] = (results_df['true_label'] == results_df['pred_label'])
    plot_focused_confusion_matrix(
        y_true=all_true_labels,
        y_pred=all_preds,
        class_names=np.linspace(0,229,230).astype(int).tolist(),
        num_classes_to_plot=8
    )
    accuracy = accuracy_score(results_df['true_label'], results_df['pred_label'])
    weighted_f1 = f1_score(results_df['true_label'], results_df['pred_label'], average='weighted', zero_division=0)
    macro_f1 = f1_score(results_df['true_label'], results_df['pred_label'], average='macro', zero_division=0)
    avg_uncertainty_correct = results_df[results_df['correct'] == True]['uncertainty'].mean()
    avg_uncertainty_incorrect = results_df[results_df['correct'] == False]['uncertainty'].mean()

    ece = 0.0
    num_bins = 10
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    for i in range(num_bins):
        in_bin = (results_df['confidence'] > bin_boundaries[i]) & (results_df['confidence'] <= bin_boundaries[i+1])
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            accuracy_in_bin = results_df[in_bin]['correct'].mean()
            avg_confidence_in_bin = results_df[in_bin]['confidence'].mean()
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin
    
    num_classes = all_mean_probs.shape[1]
    true_labels_one_hot = np.eye(num_classes)[all_true_labels]
    brier_score = np.mean(np.sum((all_mean_probs - true_labels_one_hot)**2, axis=1))


    metrics = {
        'Accuracy': accuracy,
        'Weighted F1': weighted_f1,
        'Macro F1': macro_f1,
        'Expected Calibration Error (ECE)': ece,
        'Brier Score': brier_score,
        'Avg Uncertainty (Correct)': avg_uncertainty_correct,
        'Avg Uncertainty (Incorrect)': avg_uncertainty_incorrect,
    }

    return metrics, results_df

def predict_single_sample_with_mcdropout(model, xrd_sample, phys_sample, T=50, device='cpu'):

    model.to(device)
    model.eval()      
    enable_dropout(model) 

    xrd_sample = xrd_sample.to(device)
    phys_sample = phys_sample.to(device)

    all_probs = []
    with torch.no_grad():
        for _ in range(T):
            outputs = model(xrd_sample, phys_sample)
            probs = F.softmax(outputs, dim=1).cpu().numpy()
            all_probs.append(probs)

    all_probs = np.vstack(all_probs)

    mean_probs = np.mean(all_probs, axis=0)

    top_5_indices = np.argsort(mean_probs)[-5:][::-1]

    top_5_spg = []
    top_5_mean = []
    top_5_std = []
    for i in top_5_indices:
        class_probs = all_probs[:, i]
        top_5_spg.append(i+1)
        top_5_mean.append(class_probs.mean())
        top_5_std.append(class_probs.var())
        
    return top_5_spg, top_5_mean, top_5_std

def plot_uncertainty_distribution(df):
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='uncertainty', hue='correct', stat='density', common_norm=False, kde=True)
    plt.title('Uncertainty Distribution for Correct vs. Incorrect Predictions')
    plt.xlabel('Predictive Entropy (Uncertainty)')
    plt.ylabel('Density')
    plt.grid(True)
    plt.legend(title='Prediction Correct?', labels=['Yes', 'No'])
    plt.savefig('uncertainty_distribution.png')

def plot_reliability_diagram(df):
    num_bins = 10
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    accuracies = []
    avg_confidences = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (df['confidence'] > bin_lower) & (df['confidence'] <= bin_upper)
        if in_bin.sum() > 0:
            accuracies.append(df[in_bin]['correct'].mean())
            avg_confidences.append(df[in_bin]['confidence'].mean())
        else:
            accuracies.append(0)
            avg_confidences.append(0)
            
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
    plt.plot(avg_confidences, accuracies, marker='o', linestyle='-', color='blue', label='Model Calibration')
    plt.title('Reliability Diagram')
    plt.xlabel('Average Confidence')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    plt.savefig('reliability_diagram.png')

def plot_risk_coverage_curve(df):

    df_sorted = df.sort_values(by='uncertainty', ascending=True)
    
    total_samples = len(df_sorted)
    risks = []
    coverages = []

    for i in range(1, total_samples + 1):
        subset = df_sorted.iloc[:i]
        
        coverage = i / total_samples
        
        num_errors = (subset['correct'] == False).sum()
        risk = num_errors / i
        
        coverages.append(coverage * 100) 
        risks.append(risk * 100)      

    plt.figure(figsize=(10, 6))
    plt.plot(coverages, risks, marker='.', linestyle='-')
    plt.title('Risk-Coverage Curve')
    plt.xlabel('Coverage (%) - Percentage of Most Certain Predictions Accepted')
    plt.ylabel('Risk (%) - Error Rate on Accepted Predictions')
    plt.grid(True)
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    random_risk = (1 - df['correct'].mean()) * 100
    plt.axhline(y=random_risk, color='r', linestyle='--', label=f'Overall Error Rate ({random_risk:.2f}%)')
    plt.legend()
    plt.gca().invert_xaxis() 
    plt.savefig('risk_coverage_curve.png')

def create_error_bar_chart(labels, means, std_devs, save_path=None):

    if not (len(labels) == len(means) == len(std_devs)):
        raise ValueError("Labels, means, and std_devs must have the same length.")

    title=' classification results'
    ylabel='porbability'
    xlabel='spacegroup'
    save_path='prediction_result.png'
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({'font.size': 14, 'font.family': 'sans-serif'})

    fig, ax = plt.subplots(figsize=(10, 7))
    x_pos = np.arange(len(labels))

    bars = ax.bar(
        x_pos,
        means,
        yerr=std_devs,  
        align='center',
        alpha=0.75,    
        color='royalblue',
        ecolor='black',  
        capsize=8      
    )

    ax.set_ylabel(ylabel, fontsize=16, labelpad=15)
    ax.set_xlabel(xlabel, fontsize=16, labelpad=15)
    ax.set_title(title, fontsize=18, pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha='right') 
    # ax.set_ylim(0, max(means) * 1.2) 
    ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=.25)
    ax.bar_label(bars, fmt='{:.3f}', padding=3, fontsize=10)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"saved: {save_path}")
    else:
        plt.show()

def plot_focused_confusion_matrix(y_true, y_pred, class_names,
                                  min_samples=8, max_samples=200, num_classes_to_plot=8,
                                  save_path="focused_confusion_matrix.png"):
    
    class_counts = pd.Series(y_true).value_counts()
    medium_classes = class_counts[
        (class_counts >= min_samples) & (class_counts <= max_samples)
    ]
    
    if len(medium_classes) < num_classes_to_plot:
        print(f" num of [{min_samples}, {max_samples}] < {num_classes_to_plot} ")
        selected_indices = class_counts.head(num_classes_to_plot).index.tolist()
    else:
        selected_indices = np.random.choice(medium_classes.index, num_classes_to_plot, replace=False).tolist()
    selected_indices.sort()
    print(f"choose classes: {selected_indices}")
    selected_class_names = [class_names[i] for i in selected_indices]
    
    mask = np.isin(y_true, selected_indices) | np.isin(y_pred, selected_indices)
    y_true_focused = y_true[mask]
    y_pred_focused = y_pred[mask]
    
    cm = confusion_matrix(y_true_focused, y_pred_focused, labels=selected_indices)
    
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized) 

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_normalized, 
                annot=True,       
                fmt='.2%',        
                cmap='Blues',       
                xticklabels=selected_class_names,
                yticklabels=selected_class_names)
    
    plt.title(f'part of confusion_matrix ', fontsize=16)
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"saved: {save_path}")
    else:
        plt.show()

def analyze_and_plot_after_rejection(results_df, rejection_percentage=20):

    
    initial_accuracy = (results_df['true_label'] == results_df['pred_label']).mean()
    print(f"cover 100%: {initial_accuracy:.2%}")

    coverage_rate = 1.0 - (rejection_percentage / 100.0)
    uncertainty_threshold = results_df['uncertainty'].quantile(coverage_rate)
    
    accepted_df = results_df[results_df['uncertainty'] <= uncertainty_threshold]
    
    print(f"{len(accepted_df)} / {len(results_df)} , cover ≈ {coverage_rate:.0%})")
    
    accepted_accuracy = (accepted_df['true_label'] == accepted_df['pred_label']).mean()
    print(f"cover {coverage_rate:.0%}): {accepted_accuracy:.2%}")

    y_true_accepted = accepted_df['true_label'].values
    y_pred_accepted = accepted_df['pred_label'].values
    

    new_title = f'reject {rejection_percentage}% '
    new_save_path = f"cm_after_{rejection_percentage}pct_rejection.png"

    plot_focused_confusion_matrix(
        y_true=y_true_accepted,
        y_pred=y_pred_accepted,
        class_names=np.linspace(0,229,230).astype(int).tolist(),
        save_path="rejected_focused_confusion_matrix.png"
    )

def plot_cm_for_specific_classes(y_true, y_pred, class_names, classes_to_plot, title, save_path):

    selected_indices = classes_to_plot
    selected_class_names = [class_names[i] for i in selected_indices]

    mask = np.isin(y_true, selected_indices) | np.isin(y_pred, selected_indices)
    y_true_focused, y_pred_focused = y_true[mask], y_pred[mask]

    cm = confusion_matrix(y_true_focused, y_pred_focused, labels=selected_indices)
    cm_normalized = np.nan_to_num(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis])

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=selected_class_names, yticklabels=selected_class_names,
                cbar_kws={'label': 'Recall'})
    plt.title(title, fontsize=16)
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"saved: {save_path}")
    else:
        plt.show()

def compare_cm_before_and_after_rejection(results_df, rejection_percentage=20,
                                          min_samples=8, max_samples=200, num_classes_to_plot=8):

    class_counts = pd.Series(results_df['true_label']).value_counts()
    medium_classes = class_counts[(class_counts >= min_samples) & (class_counts <= max_samples)]
    if len(medium_classes) < num_classes_to_plot:
        focused_class_indices = class_counts.head(num_classes_to_plot).index.tolist()
    else:
        focused_class_indices = np.random.choice(medium_classes.index, num_classes_to_plot, replace=False).tolist()
    focused_class_indices.sort()

    plot_cm_for_specific_classes(
        y_true=results_df['true_label'].values,
        y_pred=results_df['pred_label'].values,
        class_names=np.linspace(0,229,230).astype(int).tolist(),
        classes_to_plot=focused_class_indices, 
        title='a. part of confusionmatrix (coverage 100%)',
        save_path='cm_before_rejection.png'
    )
    
    coverage_rate = 1.0 - (rejection_percentage / 100.0)
    uncertainty_threshold = results_df['uncertainty'].quantile(coverage_rate)
    accepted_df = results_df[results_df['uncertainty'] <= uncertainty_threshold]
    
    plot_cm_for_specific_classes(
        y_true=accepted_df['true_label'].values,
        y_pred=accepted_df['pred_label'].values,
        class_names=np.linspace(0,229,230).astype(int).tolist(),
        classes_to_plot=focused_class_indices, 
        title=f'b. part of confusionmatrix (coverage {coverage_rate:.0%})',
        save_path=f'cm_after_{rejection_percentage}pct_rejection.png'
    )

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader, label_weights = get_experiment_dataloader(num_workers=20)

    model = PhyNetCNN.Model().to(device)
    model.load_state_dict(torch.load('training_results/exp_results/exp_test4_model.pth'))

    # @dataclass
    # class Args:
    #     task:str = "spg"
    # model = NoPoolCNN.Model(Args()).to(device)
    # model.load_state_dict(torch.load('training_results/baseline_results/exp_cnn_model.pth'))

    metrics, results_df = evaluate_with_uncertainty(model, test_loader, device, T=50)
    print("\n--- Uncertainty Evaluation Metrics ---")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    # analyze_and_plot_after_rejection(results_df)
    compare_cm_before_and_after_rejection(results_df)

    # data_list = list(test_loader)
    # random_batch = random.choice(data_list)
    # inputs_dict,labels = random_batch
    # top_5_spg, top_5_mean, top_5_std = predict_single_sample_with_mcdropout(model,inputs_dict['raw_xrd'],inputs_dict['physical_features'])
    # create_error_bar_chart(top_5_spg, top_5_mean, top_5_std, save_path='top_5_prediction.png')


    # plot_uncertainty_distribution(results_df)
    # plot_reliability_diagram(results_df)
    # plot_risk_coverage_curve(results_df)
