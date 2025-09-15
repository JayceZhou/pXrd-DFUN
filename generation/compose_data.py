import numpy as np
import os
import glob
from tqdm import tqdm

def get_simxrd_d_grid():
    d_grid = np.linspace(0.889, 17.659, 5000)
    return d_grid

def xrd2di(intensities):
    simxrd_d_grid = get_simxrd_d_grid()
    
    wavelength = (1.54056 * 2 + 1.54439) / 3
    # print(angles_2theta[i])
    angles_2theta = np.arange(5, 120, 0.01)
    theta_rad = np.radians(angles_2theta / 2)
    d_values = wavelength / (2 * np.sin(theta_rad))

    valid_mask = (np.isfinite(d_values) & 
                    (d_values >= simxrd_d_grid.min()) & 
                    (d_values <= simxrd_d_grid.max()))
    d_valid = d_values[valid_mask]
    i_valid = intensities[valid_mask]

    pad = np.min(simxrd_d_grid)
    interpolated = np.interp(simxrd_d_grid, d_valid[::-1], i_valid[::-1], left=pad, right=pad)
    interpolated = np.maximum(interpolated, 0)

    if np.max(interpolated) > 0:
        interpolated = np.sqrt(interpolated)
        interpolated = 100* (interpolated - np.min(interpolated)) / (np.max(interpolated) - np.min(interpolated))
    
    return interpolated

def consolidate_data(data_directory, output_file):
    
    search_path = os.path.join(data_directory, '**', '*.npy')
    file_list = glob.glob(search_path, recursive=True)

    if not file_list:
        print("error")
        return

    all_names = []
    all_features = []
    all_labels7 = []
    all_labels230 = []

    for file_path in tqdm(file_list, desc="loading"):
        try:
            all_names.append(os.path.basename(file_path)[:-4])
            data_dict = np.load(file_path, allow_pickle=True).item()
            features = xrd2di(data_dict['features'])
            all_features.append(features)
            all_labels7.append(data_dict['labels7'])
            all_labels230.append(data_dict['labels230'])
        except Exception as e:
            print(f"\n {file_path} error: {e}")
            continue
            
    
    names_array = np.array(all_names, dtype='<U30') 
    features_array = np.array(all_features, dtype=np.float32) 
    labels7_array = np.array(all_labels7, dtype=np.int16)
    labels230_array = np.array(all_labels230, dtype=np.int16)

    np.savez_compressed(output_file, 
                        names=names_array, 
                        features=features_array, 
                        labels7=labels7_array, 
                        labels230=labels230_array)
    

if __name__ == '__main__':
    DATA_DIRECTORY = 'generation/Generated/XRDs_from_ASE'
    OUTPUT_NPZ_FILE = 'simulated_xrd_7.npz' 

    consolidate_data(DATA_DIRECTORY, OUTPUT_NPZ_FILE)