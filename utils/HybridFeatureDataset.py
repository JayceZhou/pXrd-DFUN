import collections
import torch
from torch.utils.data import Dataset
import numpy as np
from scipy.signal import find_peaks, peak_widths
from scipy.stats import skew, kurtosis

def extract_physical_features(d_spacing_grid, intensity_array, num_peaks=10):

    peaks, properties = find_peaks(intensity_array, prominence=np.max(intensity_array) * 0.05)
    # peaks, properties = find_peaks(intensity_array, height=np.mean(intensity_array))
    

    widths, _, left_ips, right_ips = peak_widths(intensity_array, peaks, rel_height=0.5)

    # peak_intensities = properties['peak_heights']
    peak_intensities = properties['prominences']
    sorted_indices = np.argsort(peak_intensities)[::-1] 

    peak_features = []
    for i in range(min(num_peaks, len(sorted_indices))):
        idx = sorted_indices[i]
        peak_idx = peaks[idx]

        peak_top_d = d_spacing_grid[peak_idx]
        left_base_d = d_spacing_grid[int(left_ips[idx])]
        right_base_d = d_spacing_grid[int(right_ips[idx])]
        asymmetry = (peak_top_d - left_base_d) - (right_base_d - peak_top_d)

        d_val = d_spacing_grid[peak_idx]
        intensity_val = intensity_array[peak_idx]
        fwhm_val = widths[idx] * (d_spacing_grid[1] - d_spacing_grid[0]) 
        
        peak_features.extend([d_val, intensity_val, fwhm_val, asymmetry])

    padding_length = num_peaks * 4 - len(peak_features)
    if padding_length > 0:
        peak_features.extend([0] * padding_length)

    mean_intensity = np.mean(intensity_array)
    std_intensity = np.std(intensity_array)
    skew_intensity = skew(intensity_array)
    kurt_intensity = kurtosis(intensity_array)
    safe_intensity = np.maximum(intensity_array, 0)
    if np.sum(safe_intensity) > 0:
        centroid = np.sum(d_spacing_grid * safe_intensity) / np.sum(safe_intensity)
    else:
        centroid = 0
    statistical_features = [mean_intensity, std_intensity, skew_intensity, kurt_intensity, centroid]
    final_feature_vector = np.array(peak_features + statistical_features)
    
    return final_feature_vector

class HybridFeatureDataset(Dataset):

    def __init__(self, original_dataset, d_spacing_grid, num_peaks=10,transform = None):

        self.original_dataset = original_dataset
        self.d_spacing_grid = d_spacing_grid
        self.num_peaks = num_peaks
        self.transform = transform
        self.labels = self.original_dataset.dataset.labels-1
        class_counts = collections.Counter(self.labels)
        self.few_shot_classes = {label for label, count in class_counts.items() if count <= 3}

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        original_item = self.original_dataset[idx]
        intensity_tensor = original_item['intensity']
        label = original_item['spg'] - 1 

        # intensity_numpy = intensity_tensor[0,:].numpy()
        intensity_numpy = intensity_tensor.numpy()

        physical_features_numpy = extract_physical_features(
            self.d_spacing_grid, 
            intensity_numpy,
            self.num_peaks
        )
        
        physical_features_tensor = torch.from_numpy(physical_features_numpy)

        # if self.transform:
        #     intensity_tensor = self.transform.forward(intensity_tensor)

        if self.transform and label in self.few_shot_classes:
            intensity_tensor = self.transform.forward(intensity_tensor)
        
        inputs = {
            'raw_xrd': intensity_tensor.float(),
            'physical_features': physical_features_tensor.float()
        }
        
        return inputs, label