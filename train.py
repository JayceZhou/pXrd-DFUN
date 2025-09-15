from dataclasses import dataclass
import json
import os
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import random_split
from models import NewNet, PatchTST, PhyNetCNN
from models import NoPoolCNN
from models.modules.losses import edl_digamma_loss
from utils.ASEDataset import ASEDataset
from utils.CristalDataset import CristalDataset
from utils.HybridFeatureDataset import HybridFeatureDataset
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau, OneCycleLR
from torch.utils.data.sampler import WeightedRandomSampler
from models.modules.FocalLoss_LabelSmoothing import FocalLossWithLabelSmoothing
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from utils.SimDataset import SimDataset
from utils.data_augmentation import AugmentedDataset, data_augmentation
from utils.dataload_spg import get_data_label_spg
from collections import Counter


def one_hot_embedding(labels, num_classes=10):
    # Convert to One Hot Encoding
    y = torch.eye(num_classes).to(labels.device)
    return y[labels]

def get_class_balanced_weights(labels, num_classes=230, beta=0.9):
    class_counts = np.bincount(labels, minlength=num_classes)
    effective_num = 1.0 - np.power(beta, class_counts)
    weights = (1.0 - beta) / (effective_num + 1e-9)
    weights[class_counts == 0] = 0
    num_present_classes = np.sum(class_counts > 0)
    if np.sum(weights) > 0:
        weights = weights / np.sum(weights) * num_present_classes
    else:
        weights = np.ones(num_classes)
    return torch.tensor(weights).float()

def train_one_epoch(model, data_loader, criterion, optimizer, device, epoch_num,label_weights, uncertainty=False):
    model.train()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    progress_bar = tqdm(data_loader, desc="Training", leave=False)
    for inputs_dict, labels in progress_bar:
        raw_xrd = inputs_dict['raw_xrd'].to(device)
        phys_feats = inputs_dict['physical_features'].to(device)
        labels = labels.to(device)
        label_weights = label_weights.to(device)

        optimizer.zero_grad()
        outputs = model(raw_xrd, phys_feats)
        # outputs = model(raw_xrd)
        if uncertainty:
            batch_weights = label_weights[labels]
            # print(batch_weights)
            y_onehot = one_hot_embedding(labels, 230).to(device)
            loss = criterion(
                outputs, 
                y_onehot, 
                epoch_num, 
                230, 
                100,
                batch_weights,
                device
            )
            
            _, predicted = torch.max(outputs, 1)
        else:
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs, 1)
        loss.backward()
        optimizer.step()
        

        total_loss += loss.item() * raw_xrd.size(0)
        
        correct_predictions += (predicted == labels).sum().item()
        total_samples += labels.size(0)
        
        progress_bar.set_postfix(loss=loss.item())

    epoch_loss = total_loss / total_samples
    epoch_acc = correct_predictions / total_samples
    return epoch_loss, epoch_acc

def evaluate(model, data_loader, criterion, device, epoch_num, label_weights, uncertainty=False):
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    total_uncertainty = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs_dict, labels in data_loader:
            raw_xrd = inputs_dict['raw_xrd'].to(device)
            phys_feats = inputs_dict['physical_features'].to(device)
            labels = labels.to(device)
            label_weights = label_weights.to(device)

            outputs = model(raw_xrd, phys_feats)
            # outputs = model(raw_xrd)
            if uncertainty:
                batch_weights = label_weights[labels]
                y_onehot = one_hot_embedding(labels, 230).to(device)
                evidence = F.softplus(outputs)
                alpha = evidence + 1
                loss = criterion(
                    evidence, 
                    y_onehot, 
                    epoch_num, 
                    230, 
                    100,
                    batch_weights,
                    device
                )
                _, predicted = torch.max(evidence, 1)

                sig_uncertainty = 230 / torch.sum(alpha, dim=1)
                total_uncertainty += sig_uncertainty.sum().item()
            else:
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs, 1)

            total_loss += loss.item() * raw_xrd.size(0)
            
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = total_loss / total_samples
    epoch_acc = correct_predictions / total_samples
    avg_uncertainty = total_uncertainty / total_samples if uncertainty else 0.0
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    # scheduler.step(epoch_loss)
    return epoch_loss, epoch_acc, weighted_f1, macro_f1, avg_uncertainty

def get_simxrd_d_grid():
    d_grid = np.linspace(0.889, 17.659, 5000)
    return d_grid

# load simulated data
def get_dataloader(datapath='simulated_xrd.npz', batch_size=32, num_workers=8):
    dataset = SimDataset(datapath)
    labels = dataset.labels
    indices = list(range(len(dataset)))
    train_indices, test_indices = train_test_split(
        indices,
        test_size=0.2,       
        # stratify=labels,     
        random_state=42     
    )
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    d_grid = np.flip(get_simxrd_d_grid())

    hybrid_train_dataset = HybridFeatureDataset(train_dataset, d_grid)
    hybrid_test_dataset = HybridFeatureDataset(test_dataset, d_grid)

    train_loader = DataLoader(hybrid_train_dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    test_loader = DataLoader(hybrid_test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader


def augment_one_shot_classes(intensities, labels, length=5000):
    augment = data_augmentation(settings = (0, 1, length, 230, 'cuda:1', 0.8, 32), 
                                settings_aug = (50, 10, 0.1, 0.2, 0.2))
    class_counts = Counter(labels)
    one_shot_classes = {label for label, count in class_counts.items() if count == 1}
    
    if not one_shot_classes:
        return intensities, labels
    
    new_intensities = []
    new_labels = []

    for i in range(len(labels)):
        label = labels[i]
        if label in one_shot_classes:
            original_intensity = torch.tensor(intensities[i], dtype=torch.float32)
            
            augmented_intensity = augment.forward(original_intensity)
            augmented_intensity = np.clip(augmented_intensity, 0, None)
            
            new_intensities.append(augmented_intensity)
            new_labels.append(label)
            
    if new_intensities:
        new_intensities_np = np.array(new_intensities, dtype=intensities.dtype)
        new_labels_np = np.array(new_labels, dtype=labels.dtype)
        
        augmented_intensities = np.concatenate((intensities, new_intensities_np), axis=0)
        augmented_labels = np.concatenate((labels, new_labels_np), axis=0)
        
    else:
        augmented_intensities, augmented_labels = intensities, labels

    return augmented_intensities, augmented_labels

def get_experiment_dataloader(data_path='experiment_xrd.npz', batch_size=32 , num_workers=8):
    # d_grid, intensities, label = get_data_label_spg(data_path)
    d_grid = np.flip(get_simxrd_d_grid())
    xrd_datasets = np.load(data_path)
    intensities = xrd_datasets['features']
    labels = xrd_datasets['labels230']
    intensities, labels = augment_one_shot_classes(intensities, labels, 5000)
    label_weights = get_class_balanced_weights(labels)
    dataset = CristalDataset(intensities, labels)
    indices = list(range(len(dataset)))
    train_indices, test_indices = train_test_split(
        indices,
        test_size=0.2,       
        stratify=dataset.labels,     
        random_state=42     
    )
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)
    hybrid_train_dataset = HybridFeatureDataset(train_dataset, d_grid)
    # hybrid_train_dataset = HybridFeatureDataset(train_dataset, d_grid, 
    #                                             transform=data_augmentation(settings = (train_indices, 1, 5000, 230, 'cuda', 0.8, batch_size), 
    #                                                                         settings_aug = (50, 10, 0.1, 0.2, 0.2)))
    hybrid_test_dataset = HybridFeatureDataset(test_dataset, d_grid)

    train_labels = labels[train_indices]
    class_counts = np.bincount(train_labels, minlength=230)
    class_weights = 1. / (class_counts+1e-9)
    sample_weights = np.array([class_weights[t] for t in train_labels])
    sample_weights = torch.from_numpy(sample_weights).double()
    
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    
    train_loader = DataLoader(hybrid_train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(hybrid_test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader,test_loader,label_weights

def get_rruff_dataloader(data_path='rruff_xrd.npz', batch_size=32 , num_workers=8):
    # d_grid, intensities, label = get_data_label_spg(data_path)
    d_grid = np.linspace(5,90,8500)
    xrd_datasets = np.load(data_path)
    intensities = xrd_datasets['features']
    label = xrd_datasets['labels230']
    intensities, label = augment_one_shot_classes(intensities, label, 8500)
    dataset = CristalDataset(intensities, label)
    indices = list(range(len(dataset)))
    train_indices, test_indices = train_test_split(
        indices,
        test_size=0.2,       
        stratify=dataset.labels,     
        random_state=42     
    )
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)
    # hybrid_train_dataset = HybridFeatureDataset(train_dataset, d_grid)
    hybrid_train_dataset = HybridFeatureDataset(train_dataset, d_grid, 
                                                transform=data_augmentation(settings = (train_indices, 1, 8500, 230, 'cuda', 0.8, batch_size), 
                                                                            settings_aug = (50, 5, 0.1, 0.2, 0.2)))
    hybrid_test_dataset = HybridFeatureDataset(test_dataset, d_grid)
    
    train_loader = DataLoader(hybrid_train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(hybrid_test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader,test_loader

def plot_history(history, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['test_loss'], label='Test Loss')
    ax1.set_title('Loss curve')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history['train_acc'], label='Train Accuracy')
    ax2.plot(history['test_acc'], label='Test Accuracy')
    ax2.set_title('Accuracy Curve')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    CONFIG = {
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "EPOCHS": 300,
    "LEARNING_RATE": 5e-4,
    "UNCERTAINTY": False, 
    }

    label_weights = torch.tensor(1)
    # train_loader, test_loader = get_dataloader(datapath='simulated_xrd_7.npz', num_workers=20)
    train_loader, test_loader, label_weights = get_experiment_dataloader(num_workers=20)
    # train_loader, test_loader = get_rruff_dataloader(num_workers=20)

    # model = PhyNetCNN.Model().to(CONFIG['DEVICE'])
    @dataclass
    class Args:
        task:str = "spg"
        
    
    # model = NoPoolCNN.Model(Args()).to(CONFIG['DEVICE'])
    # model = NewNet.Model(Args()).to(CONFIG['DEVICE'])
    # model = PatchTST.Model(Args()).to(CONFIG['DEVICE'])
    model = PhyNetCNN.Model().to(CONFIG['DEVICE'])


    criterion = nn.CrossEntropyLoss()
    
    # criterion = FocalLossWithLabelSmoothing()
    # criterion = StandardFocalLoss(gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['LEARNING_RATE'], weight_decay=1e-3)
    # scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=2, verbose=True)

    SAVE_DIR = "./training_results/ablation"
    os.makedirs(SAVE_DIR, exist_ok=True)
    HISTORY_PATH = os.path.join(SAVE_DIR, "ablation4_acc.json")

    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'weighted_f1': [],
        'test_uncertainty': []
    }
    best_test_loss = float('inf')
    best_test_acc = 0.0

    print("device:", CONFIG['DEVICE'])
    early_stop = 0
    # total_steps = len(train_loader) * CONFIG['EPOCHS']

    # scheduler = OneCycleLR(optimizer, max_lr=5e-4, total_steps=total_steps)
    scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG['EPOCHS'], eta_min=1e-7)
    for epoch in range(CONFIG['EPOCHS']):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, CONFIG['DEVICE'], epoch,label_weights, CONFIG['UNCERTAINTY'])
        scheduler.step()
        test_loss, test_acc, weighted_f1, macro_f1, avg_uncertainty = evaluate(model, test_loader, criterion, CONFIG['DEVICE'], epoch,label_weights, CONFIG['UNCERTAINTY'])

        print(
            f"Epoch {epoch+1}/{CONFIG['EPOCHS']} | "
            f"train loss: {train_loss:.4f} | train acc: {train_acc:.4f} | "
            f"test loss: {test_loss:.4f} | test acc: {test_acc:.4f} | "
            f"Weighted F1: {weighted_f1:.4f} | "
            f"Macro F1: {macro_f1:.4f} | "
        )
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['weighted_f1'].append(weighted_f1)
        # history['test_uncertainty'].append(avg_uncertainty)
        
        early_stop += 1
        with open(HISTORY_PATH, 'w') as f:
            json.dump(history, f, indent=4)
        if test_loss < best_test_loss:
            early_stop = 0
            best_test_loss = test_loss
            best_model_path = os.path.join(SAVE_DIR, "ablation4_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"    -> saved: {best_model_path}")
        # if test_acc > best_test_acc:
        #     early_stop = 0
        #     best_test_acc = test_acc
        #     best_model_path = os.path.join(SAVE_DIR, "ablation4_model.pth")
        #     torch.save(model.state_dict(), best_model_path)
        #     print(f"    -> saved: {best_model_path}")
        if early_stop >= 5:
            break

    plot_path = os.path.join(SAVE_DIR, "ablation4_curves.png")
    plot_history(history, plot_path)