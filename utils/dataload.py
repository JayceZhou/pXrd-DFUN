import os
import json
import numpy as np
from scipy.interpolate import interp1d

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

    # lengths = list(map(float, level2['lengths']))
    # angles = list(map(float, level2['angles']))

    # print("Unit Cell Parameters:")
    # print(f"Lengths (a, b, c): {lengths}")
    # print(f"Angles (α, β, γ): {angles}")
    # print("Atomic positions (fractional):")
    for i, atom in enumerate(atoms):
        atom = json.loads(atom)  # 解析每个原子信息
        new_atoms.append(atom)
        # print(f"Atom {i+1}: {atom['symbol']} at ({atom['x']}, {atom['y']}, {atom['z']}), Occ={atom['occupancy']}")

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
    
def get_xrd_data(crystal_data):
    theta = crystal_data[:,0]
    intensity = crystal_data[:,1]
    primary_wavelength = np.array(
        [d['primary_wavelength'] for d in [d['xray_info'] 
                                for d in crystal_data[:,2]]],dtype=float)
    secondary_wavelength = np.array(
        [d['secondary_wavelength'] for d in [d['xray_info'] 
                                for d in crystal_data[:,2]]],dtype=object)
    label = label2cristal_system(crystal_data)
    return theta, intensity, primary_wavelength, secondary_wavelength, label

def label2cristal_system(crystal_data):
    label1 = np.array([d['spacegroup'] 
                       for d in crystal_data[:,2]],dtype=int)
    label2 = np.array([d['crystal_system'] 
                       for d in crystal_data[:,2]],dtype=object)
    for i in range(len(crystal_data)):
        if label1[i] <= 2 and label2[i] == None:
            label2[i] = 'triclinic'
        elif label1[i] <= 15 and label2[i] == None:
            label2[i] = 'monoclinic'
        elif label1[i] <= 74 and label2[i] == None:
            label2[i] = 'orthorhombic'
        elif label1[i] <= 142 and label2[i] == None:
            label2[i] = 'tetragonal'
        elif label1[i] <= 167 and label2[i] == None:
            label2[i] = 'trigonal'
        elif label1[i] <= 194 and label2[i] == None:
            label2[i] = 'hexagonal'
        elif label1[i] <= 230 and label2[i] == None:
            label2[i] = 'cubic'
    return label2

def get_simxrd_d_grid(wavelength=1.5406):
    angles_2theta_simxrd = np.linspace(10, 80, 3501)
    theta_rad = np.radians(angles_2theta_simxrd / 2)
    d_grid = wavelength / (2 * np.sin(theta_rad))
    return d_grid[::-1]

def xrd2dI(crystal_data):
    
    angles_2theta, intensities, primary_wavelength, secondary_wavelength, label = get_xrd_data(crystal_data)
    
    simxrd_d_grid = get_simxrd_d_grid()
    N = intensities.shape[0]
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
            label[i] = None
            continue

        interpolated = np.interp(simxrd_d_grid, d_valid[::-1], i_valid[::-1])

        if np.max(interpolated) > 0:
            interpolated *= 100 / np.max(interpolated)

        batch_dI[i] = interpolated
    
    batch_dI = batch_dI[~np.all(batch_dI == 0, axis=1)]
    new_label = label[label != None]

    return simxrd_d_grid,batch_dI,new_label

def get_data_label(base_dir):
    data_files = read_all_json_files(base_dir)
    crystal_data = load_data(data_files)
    d_grid, intensities, label = xrd2dI(crystal_data)
    return d_grid, intensities, label

if __name__ == '__main__':
    base_dir = 'data'
    data_files = read_all_json_files(base_dir)
    crystal_data = load_data(data_files)
    theta, intensity, primary_wavelength, secondary_wavelength = get_xrd_data(crystal_data)
    d_grid, intensities = xrd2dI(theta, intensity, primary_wavelength, secondary_wavelength)
    print(len(d_grid), len(intensities))
    