import os
import json
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

def read_all_json_files(base_dir):
    json_data_list = []

    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.json'):
                json_path = os.path.join(root, file)
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        json_data_list.append((json_path, data))
                except Exception as e:
                    print(f"Failed to read {json_path}: {e}")
    
    return json_data_list

def label2dict(raw_label):

    level1 = json.loads(raw_label)

    phases_raw = level1['phases'][0]  # 字符串
    level2 = json.loads(phases_raw)

    atoms_raw = level2['base']
    atoms = json.loads(atoms_raw)
    new_atoms = []

    for i, atom in enumerate(atoms):
        atom = json.loads(atom)  # 解析每个原子信息
        new_atoms.append(atom)

    label = {}
    for key in json.loads(raw_label).keys():
        if key != 'phases':
            label[key] = json.loads(raw_label)[key]
            # print(f"{key}: {label[key]}")
    for key in level2.keys():
        if key != 'base':
            label[key] = level2[key]
            # print(f"{key}: {level2[key]}")

    label['xray_info'] = json.loads(label['xray_info'])

    return label, new_atoms

def load_data(data_files):
    dataset = []
    for _,data in data_files:
        label, atoms = label2dict(data['label'])
        x = np.array(data['two_theta_values'])
        y = np.array(data['intensities'])
        dataset.append(( x, y, label, atoms))

    crystal_data = []
    for data in dataset:
        label = data[2]
        if label['spacegroup'] != None :
            xray_info = label['xray_info']
            if not (xray_info['primary_wavelength'] is None):
                crystal_data.append(data)
    crystal_data = np.array(crystal_data,dtype=object)

    return crystal_data
    
def get_xrd_data_spg(crystal_data):
    theta = crystal_data[:,0]
    intensity = crystal_data[:,1]
    primary_wavelength = np.array(
        [d['primary_wavelength'] for d in [d['xray_info'] 
                                for d in crystal_data[:,2]]],dtype=float)
    secondary_wavelength = np.array(
        [d['secondary_wavelength'] for d in [d['xray_info'] 
                                for d in crystal_data[:,2]]],dtype=object)
    label = np.array([d['spacegroup'] 
                       for d in crystal_data[:,2]],dtype=int)
    return theta, intensity, primary_wavelength, secondary_wavelength, label


def get_simxrd_d_grid():
    d_grid = np.linspace(0.889, 17.659, 5000)
    return d_grid


def xrd2dI_spg(crystal_data):
    
    angles_2theta, intensities, primary_wavelength, secondary_wavelength, label = get_xrd_data_spg(crystal_data)
    
    simxrd_d_grid = get_simxrd_d_grid()
    N = intensities.shape[0]
    window_length = 11
    polyorder = 3
    batch_dI = np.zeros((N, len(simxrd_d_grid)))

    for i in range(N):
        if secondary_wavelength[i] is None:
            wavelength = primary_wavelength[i]
        else:
            wavelength = (primary_wavelength[i] * 2 + 
                            float(secondary_wavelength[i])) / 3

        # print(angles_2theta[i])
        theta_rad = np.radians(angles_2theta[i] / 2)
        d_values = wavelength / (2 * np.sin(theta_rad))

        valid_mask = (np.isfinite(d_values) & 
                        (d_values >= simxrd_d_grid.min()) & 
                        (d_values <= simxrd_d_grid.max()))
        d_valid = d_values[valid_mask]
        i_valid = intensities[i][valid_mask]
        if len(d_valid) == 0 or len(i_valid) == 0:
            label[i] = -1
            continue
        inten_tmp = savgol_filter(i_valid[::-1], window_length, polyorder)
        inten_tmp = np.maximum(inten_tmp, 0)
        pad = np.min(inten_tmp)
        interpolated = np.interp(simxrd_d_grid, d_valid[::-1], inten_tmp,left=pad, right=pad)
        interpolated = np.maximum(interpolated, 0)
        if np.max(interpolated) > 0:
            interpolated = np.sqrt(interpolated)
            interpolated = 100* (interpolated - np.min(interpolated)) / (np.max(interpolated) - np.min(interpolated))

        batch_dI[i] = interpolated

    batch_dI = batch_dI[~np.all(batch_dI == 0, axis=1)]
    # batch_dI = savgol_filter(batch_dI, window_length, polyorder)
    # batch_dI = np.maximum(batch_dI, 0)
    new_label = label[label != -1]

    return simxrd_d_grid,batch_dI,new_label

def get_data_label_spg(base_dir):
    data_files = read_all_json_files(base_dir)
    crystal_data = load_data(data_files)
    d_grid, intensities, label = xrd2dI_spg(crystal_data)
    return d_grid, intensities, label