import os
import random
import re
import numpy as np
import glob
from tqdm import tqdm
from torch import as_tensor, multiprocessing as mp
import ase.db
from ase.spacegroup import get_spacegroup

WAVELENGTH_A1 = 1.54056  # Cu K-alpha1 
WAVELENGTH_A2 = 1.54439  # Cu K-alpha2 
INTENSITY_RATIO_A2_A1 = 0.5  

def apply_march_dollase(hkl_info, intensities, crystal_system, cell_lengths_and_angles):

    preferred_planes = [(0, 0, 1), (1, 0, 0), (1, 1, 0), (1, 1, 1)]
    h_p, k_p, l_p = random.choice(preferred_planes)

    r = random.uniform(0.4, 1.0)
    
    if r > 0.98:
        return intensities

    h, k, l = hkl_info[:, 0], hkl_info[:, 1], hkl_info[:, 2]
    
    norm_hkl = np.sqrt(h**2 + k**2 + l**2)
    norm_pref = np.sqrt(h_p**2 + k_p**2 + l_p**2)
    
    valid_indices = (norm_hkl > 1e-6) & (norm_pref > 1e-6)
    cos_alpha = np.ones_like(h, dtype=float) 

    dot_product = h[valid_indices] * h_p + k[valid_indices] * k_p + l[valid_indices] * l_p
    cos_alpha[valid_indices] = dot_product / (norm_hkl[valid_indices] * norm_pref)

    r_sq = r**2
    sin_alpha_sq = 1 - cos_alpha**2
    
    correction_factor = (r_sq * cos_alpha**2 + (1/r) * sin_alpha_sq)**(-1.5)

    corrected_intensities = intensities * correction_factor
    
    return corrected_intensities

def add_polynomial_background(pattern, x_axis):
    order = random.randint(3, 6)
    
    x_norm = (x_axis - x_axis[0]) / (x_axis[-1] - x_axis[0])
    
    coeffs = np.random.randn(order + 1)
    coeffs[-1] *= 0.01 
    coeffs[-2] *= 0.1 
    
    p = np.poly1d(coeffs)
    background = p(x_norm)
    
    background = background - np.min(background)
    bg_max = np.max(background)
    if bg_max > 0:
        background_height = random.uniform(2, 10)
        background = (background / bg_max) * background_height
        
    return pattern + background

def apply_poisson_noise(pattern, max_virtual_counts=10000):
    pattern[pattern < 0] = 0
    
    current_max = np.max(pattern)
    if current_max == 0:
        return pattern 
        
    scaled_pattern = pattern * (max_virtual_counts / current_max)
    noisy_scaled_pattern = np.random.poisson(scaled_pattern)
    final_pattern = noisy_scaled_pattern * (current_max / max_virtual_counts)
    
    return final_pattern

def hkl(hkl_max=10, hkl_path=None):
    hkl_info = np.array([[1, 0, 0]])
    hkl_add = np.zeros((2, 3))
    # Here we start our loop to increase hkl rows one by one sequentially
    hkl_idx = 0
    hkl_h = 1
    while hkl_h <= hkl_max:
        if hkl_info[hkl_idx, 1] == hkl_info[hkl_idx, 2] and hkl_info[hkl_idx, 0] != hkl_info[hkl_idx, 1]:
            hkl_add[0] = hkl_info[hkl_idx]
            hkl_add[0, 1] = hkl_add[0, 1] + 1
            hkl_add[0, 2] = 0
            hkl_info = np.vstack([hkl_info, hkl_add[0]])
        elif hkl_info[hkl_idx, 1] > hkl_info[hkl_idx, 2]:
            hkl_add[0] = hkl_info[hkl_idx]
            hkl_add[0, 2] = hkl_add[0, 2] + 1
            hkl_info = np.vstack([hkl_info, hkl_add[0]])
        elif hkl_info[hkl_idx, 0] == hkl_info[hkl_idx, 1] and hkl_info[hkl_idx, 0] == hkl_info[hkl_idx, 2]:
            hkl_add[0] = hkl_info[hkl_idx]
            hkl_add[0, 0] = hkl_add[0, 0] + 1
            hkl_add[0, 1] = 0
            hkl_add[0, 2] = 0
            if hkl_h != hkl_max:
                hkl_info = np.vstack([hkl_info, hkl_add[0]])
            hkl_h += 1
        hkl_idx += 1

    hkl_exp = hkl_info[0, :]

    for i in range(0, hkl_info.shape[0]):
        hkl_switch01 = np.zeros((2, 3))
        hkl_switch01[0, 0] = hkl_info[i, 1]
        hkl_switch01[0, 1] = hkl_info[i, 0]
        hkl_switch01[0, 2] = hkl_info[i, 2]
        hkl_switch12 = np.zeros((2, 3))
        hkl_switch12[0, 0] = hkl_info[i, 0]
        hkl_switch12[0, 1] = hkl_info[i, 2]
        hkl_switch12[0, 2] = hkl_info[i, 1]
        hkl_switch02 = np.zeros((2, 3))
        hkl_switch02[0, 0] = hkl_info[i, 2]
        hkl_switch02[0, 1] = hkl_info[i, 1]
        hkl_switch02[0, 2] = hkl_info[i, 0]
        hkl_displace201 = np.zeros((2, 3))
        hkl_displace201[0, 0] = hkl_info[i, 2]
        hkl_displace201[0, 1] = hkl_info[i, 0]
        hkl_displace201[0, 2] = hkl_info[i, 1]
        hkl_displace120 = np.zeros((2, 3))
        hkl_displace120[0, 0] = hkl_info[i, 1]
        hkl_displace120[0, 1] = hkl_info[i, 2]
        hkl_displace120[0, 2] = hkl_info[i, 0]
        hkl_exp = np.vstack([hkl_exp, hkl_info[i, :]])
        hkl_exp = np.vstack([hkl_exp, hkl_switch01[0]])
        hkl_exp = np.vstack([hkl_exp, hkl_switch12[0]])
        hkl_exp = np.vstack([hkl_exp, hkl_switch02[0]])
        hkl_exp = np.vstack([hkl_exp, hkl_displace201[0]])
        hkl_exp = np.vstack([hkl_exp, hkl_displace120[0]])

    # Then, reduce identical row
    hkl_redu = np.zeros((1, 3))
    hkl_redu[0] = hkl_exp[0]
    # Loop for extract
    for i in range(1, hkl_exp.shape[0]):
        # Loop for line by line comparison
        vstack_judge = True
        if_loop_judge = False
        for j in range(0, hkl_redu.shape[0]):
            if np.array_equal(hkl_exp[i], hkl_redu[j]):
                vstack_judge = False
            if_loop_judge = True
        if vstack_judge and if_loop_judge:
            hkl_redu = np.vstack([hkl_redu, hkl_exp[i]])

    # Now, we put negative signs in the matrix
    # for hkl_exp, we extract every line and then vstack to hkl_exp2\
    hkl_exp2 = hkl_redu[0, :]
    for i in range(1, hkl_redu.shape[0]):
        # 1st case: 2 0s
        if hkl_redu[i, 0] * hkl_redu[i, 1] == 0 and hkl_redu[i, 0] * hkl_redu[i, 2] == 0 \
                and hkl_redu[i, 1] * hkl_redu[i, 2] == 0:
            hkl_exp2 = np.vstack([hkl_exp2, hkl_redu[i, :]])
        # 2nd case: 1 0s
        elif hkl_redu[i, 0] == 0 or hkl_redu[i, 2] == 0 or hkl_redu[i, 1] == 0:
            hkl_one0_1 = np.zeros((2, 3))
            if hkl_redu[i, 2] == 0:
                hkl_one0_1[0, 0] = hkl_redu[i, 0]
                hkl_one0_1[0, 1] = -hkl_redu[i, 1]
                hkl_one0_1[0, 2] = hkl_redu[i, 2]
            elif hkl_redu[i, 0] == 0:
                hkl_one0_1[0, 0] = hkl_redu[i, 0]
                hkl_one0_1[0, 1] = hkl_redu[i, 1]
                hkl_one0_1[0, 2] = -hkl_redu[i, 2]
            elif hkl_redu[i, 1] == 0:
                hkl_one0_1[0, 0] = hkl_redu[i, 0]
                hkl_one0_1[0, 1] = hkl_redu[i, 1]
                hkl_one0_1[0, 2] = -hkl_redu[i, 2]
            hkl_exp2 = np.vstack([hkl_exp2, hkl_redu[i, :]])
            hkl_exp2 = np.vstack([hkl_exp2, hkl_one0_1[0, :]])
        # 3rd case: none 0
        else:
            hkl_none0_1 = np.zeros((2, 3))
            hkl_none0_2 = np.zeros((2, 3))
            hkl_none0_3 = np.zeros((2, 3))
            hkl_none0_1[0, 0] = hkl_redu[i, 0]
            hkl_none0_1[0, 1] = -hkl_redu[i, 1]
            hkl_none0_1[0, 2] = hkl_redu[i, 2]
            hkl_none0_2[0, 0] = hkl_redu[i, 0]
            hkl_none0_2[0, 1] = hkl_redu[i, 1]
            hkl_none0_2[0, 2] = -hkl_redu[i, 2]
            hkl_none0_3[0, 0] = hkl_redu[i, 0]
            hkl_none0_3[0, 1] = -hkl_redu[i, 1]
            hkl_none0_3[0, 2] = -hkl_redu[i, 2]
            hkl_exp2 = np.vstack([hkl_exp2, hkl_redu[i, :]])
            hkl_exp2 = np.vstack([hkl_exp2, hkl_none0_1[0, :]])
            hkl_exp2 = np.vstack([hkl_exp2, hkl_none0_2[0, :]])
            hkl_exp2 = np.vstack([hkl_exp2, hkl_none0_3[0, :]])

    hkl_multi = np.ones((hkl_exp2.shape[0], 1))
    hkl_final = np.hstack([hkl_exp2, hkl_multi])

    if hkl_path is None:
        hkl_path = f'./hkl_{hkl_max}.npy'

    np.save(hkl_path, hkl_final)
    return hkl_final

def gaus(x, h):
    const_g = 4 * np.log(2)
    value = ((const_g ** (1 / 2)) / (np.pi ** (1 / 2) * h)) * np.exp(-const_g * (x / h) ** 2)
    return value


def y_multi(x_val, step, xy_merge, H):
    y_val = 0
    for xy_idx in range(0, xy_merge.shape[0]):
        angle = xy_merge[xy_idx, 0]
        inten = xy_merge[xy_idx, 1]
        if abs(x_val * step - angle) < 5: 
            y_val += inten * (gaus((x_val * step - angle), H[xy_idx, 0]) * 1.5)
    return y_val

hkl_max = 10

path = f'{os.path.dirname(__file__)}/Generated'
os.makedirs(path, exist_ok=True)

hkl_path = f'{path}/hkl_{hkl_max}.npy'
if os.path.exists(hkl_path):
    if mp.current_process().name == 'MainProcess':
        print('load hkl...', end=" ")
    _hkl_info = as_tensor(np.load(hkl_path)).share_memory_()
else:
    if mp.current_process().name == 'MainProcess':
        print('compute hkl...', end=" ")
    _hkl_info = as_tensor(hkl(hkl_max, hkl_path)).share_memory_()
if mp.current_process().name == 'MainProcess':
    print('- ok✓')

space_group_map_dict = {}
for i in range(1, 3): space_group_map_dict[i] = 1    # Triclinic
for i in range(3, 16): space_group_map_dict[i] = 2   # Monoclinic
for i in range(16, 75): space_group_map_dict[i] = 3  # Orthorhombic
for i in range(75, 143): space_group_map_dict[i] = 4 # Tetragonal
for i in range(143, 168): space_group_map_dict[i] = 5# Trigonal
for i in range(168, 195): space_group_map_dict[i] = 6# Hexagonal
for i in range(195, 231): space_group_map_dict[i] = 7# Cubic

try:
    with open(os.path.dirname(__file__) + '/ION_SCATTERING_TABLE.txt', 'r') as scat_table:
        scat_table_lines = scat_table.readlines()
except FileNotFoundError:
    print("can not find 'ION_SCATTERING_TABLE.txt'。")
    exit()


def process_atoms_row(row, hkl_info=_hkl_info, x_step=0.01, save_path=None, apply_preferred_orientation=False):
    try:
        hkl_info = hkl_info.numpy()
        atoms = row.toatoms()

        if len(atoms) == 0:
            return f"Failed: ID {row.id} "

        cell_lengths_and_angles = atoms.cell.cellpar()
        cell_a, cell_b, cell_c, cell_alpha_deg, cell_beta_deg, cell_gamma_deg = cell_lengths_and_angles
        
        cell_alpha = np.deg2rad(cell_alpha_deg)
        cell_beta = np.deg2rad(cell_beta_deg)
        cell_gamma = np.deg2rad(cell_gamma_deg)

        try:
            sg = get_spacegroup(atoms, symprec=1e-5) 
            space_group = sg.no
        except Exception:
            return f"Failed: ID {row.id} "
        
        if not (1 <= space_group <= 230):
            return f"Failed: ID {row.id} "
        
        crystal_system = space_group_map_dict.get(space_group)

        atomic_numbers = atoms.get_atomic_numbers()
        scaled_positions = atoms.get_scaled_positions()
        occupancies = np.ones((len(atoms), 1))
        
        cell_info = np.hstack([
            atomic_numbers.reshape(-1, 1),
            scaled_positions,
            occupancies
        ])

        hkl_h, hkl_k, hkl_l = hkl_info[:, 0], hkl_info[:, 1], hkl_info[:, 2]
        
        a, b, c, alpha, beta, gamma = cell_a, cell_b, cell_c, cell_alpha, cell_beta, cell_gamma
        v = (a * b * c) * (1 + 2 * np.cos(alpha) * np.cos(beta) * np.cos(gamma) - np.cos(alpha)**2 - np.cos(beta)**2 - np.cos(gamma)**2)**0.5
        
        if v == 0: return f"Failed: ID {row.id}"
        
        term1 = (hkl_h**2 * b**2 * c**2 * np.sin(alpha)**2)
        term2 = (hkl_k**2 * a**2 * c**2 * np.sin(beta)**2)
        term3 = (hkl_l**2 * a**2 * b**2 * np.sin(gamma)**2)
        term4 = (2 * hkl_h * hkl_k * a * b * c**2 * (np.cos(alpha) * np.cos(beta) - np.cos(gamma)))
        term5 = (2 * hkl_k * hkl_l * a**2 * b * c * (np.cos(beta) * np.cos(gamma) - np.cos(alpha)))
        term6 = (2 * hkl_h * hkl_l * a * b**2 * c * (np.cos(alpha) * np.cos(gamma) - np.cos(beta)))
        
        d_inv_sq = (1 / v**2) * (term1 + term2 + term3 + term4 + term5 + term6)
        d_inv_sq[d_inv_sq < 0] = 0
        hkl_d = 1 / np.sqrt(d_inv_sq, out=np.full_like(d_inv_sq, np.inf), where=d_inv_sq!=0)

        wavelength = WAVELENGTH_A1  # Cu K-alpha
        ratio = wavelength / (2 * hkl_d)
        valid_indices = np.where(np.abs(ratio) <= 1)
        
        two_theta = np.zeros(hkl_info.shape[0])
        two_theta_pi = np.zeros(hkl_info.shape[0])
        
        theta_cal = np.arcsin(ratio[valid_indices])
        two_theta[valid_indices] = np.rad2deg(2 * theta_cal)
        two_theta_pi[valid_indices] = 2 * theta_cal

        hkl_2theta = np.hstack([hkl_info, two_theta.reshape(-1, 1), two_theta_pi.reshape(-1, 1)])
        hkl_2theta = hkl_2theta[hkl_2theta[:, 4] > 1e-5] 

        if hkl_2theta.shape[0] == 0:
            return f"Failed: ID {row.id} "

        hkl_info, two_theta, two_theta_pi = hkl_2theta[:, :4], hkl_2theta[:, 4], hkl_2theta[:, 5]

        lp = (1 + np.cos(two_theta_pi)**2) / (np.cos(two_theta_pi / 2) * np.sin(two_theta_pi / 2)**2)

        s = np.sin(two_theta_pi / 2) / wavelength
        atom_scat = np.zeros((hkl_info.shape[0], cell_info.shape[0]))
        for i in range(cell_info.shape[0]):
            z_index = int(cell_info[i, 0])
            if not (1 <= z_index < len(scat_table_lines)):
                return f"Failed: ID {row.id} "
            
            coeffs = np.array(scat_table_lines[z_index - 1].split()[3:12], dtype=float)
            f = coeffs[0] * np.exp(-coeffs[1] * s**2) + \
                coeffs[2] * np.exp(-coeffs[3] * s**2) + \
                coeffs[4] * np.exp(-coeffs[5] * s**2) + \
                coeffs[6] * np.exp(-coeffs[7] * s**2) + \
                coeffs[8]
            atom_scat[:, i] = f * cell_info[i, 4]  # 乘以占有率


        hkl_pos = np.dot(hkl_info[:, 0:3], cell_info[:, 1:4].T)
        
        # F_hkl = sum(f_n * exp(2*pi*i * (h*x_n + k*y_n + l*z_n)))
        exp_term = np.exp(2 * np.pi * 1j * hkl_pos)
        f_hkl = np.sum(atom_scat * exp_term, axis=1)
        
        struc = np.abs(f_hkl)**2
        intensities = lp * struc

        if apply_preferred_orientation:
            intensities = apply_march_dollase(hkl_info, intensities, crystal_system, cell_lengths_and_angles)


        x_y = np.vstack([two_theta, intensities]).T
        x_y = x_y[x_y[:, 1] > 1e-3]

        if x_y.shape[0] == 0:
            return f"Failed: ID {row.id} "
            
        xy_merge = []
        x_y = x_y[np.argsort(x_y[:, 0])] #
        
        current_angle = x_y[0, 0]
        current_intensity = x_y[0, 1]

        for i in range(1, x_y.shape[0]):
            if abs(x_y[i, 0] - current_angle) < 0.02: 
                current_intensity += x_y[i, 1]
            else:
                xy_merge.append([current_angle, current_intensity])
                current_angle = x_y[i, 0]
                current_intensity = x_y[i, 1]
        xy_merge.append([current_angle, current_intensity])
        xy_merge = np.array(xy_merge)

        peak_shapes = [(0.05, -0.06, 0.07), (0.05, -0.01, 0.01),
                       (0.0, 0.0, 0.01), (0.0, 0.0, random.uniform(0.001, 0.1))]
        # peak_shapes = [(0.05, -0.06, 0.07)]
        
        XRDs = []
        for peak_shape_idx, (U, V, W) in enumerate(peak_shapes):
            tan_theta = np.tan(np.deg2rad(xy_merge[:, 0]) / 2)
            H_sq = U * tan_theta**2 + V * tan_theta + W
            H_sq[H_sq < 0] = 1e-6 
            H = np.sqrt(H_sq).reshape(-1, 1)

            total_points = int(180 / x_step)
            pattern_x = np.arange(0, 180, x_step)
            pattern_y = np.zeros(total_points)

            wavelength_ratio = WAVELENGTH_A2 / WAVELENGTH_A1
            for i in range(xy_merge.shape[0]):
                angle_a1 = xy_merge[i, 0]
                intensity_a1 = xy_merge[i, 1]
                h_val = H[i, 0]

                pattern_y += intensity_a1 * gaus(pattern_x - angle_a1, h_val)

                sin_theta_a1 = np.sin(np.deg2rad(angle_a1) / 2)
                sin_arg = sin_theta_a1 * wavelength_ratio

                if sin_arg <= 1.0:
                    angle_a2 = np.rad2deg(2 * np.arcsin(sin_arg))
                    intensity_a2 = intensity_a1 * INTENSITY_RATIO_A2_A1
                    pattern_y += intensity_a2 * gaus(pattern_x - angle_a2, h_val)

            # for i in range(xy_merge.shape[0]):
            #     pattern_y += xy_merge[i, 1] * gaus(pattern_x - xy_merge[i, 0], H[i, 0]) * 1.5

            pattern_y /= np.max(pattern_y)
            pattern_y *= 1.5
            
            features = pattern_y[int(5/x_step):int(120/x_step)]
            max_intensity = np.max(features)
            if max_intensity > 0:
                normalized_pattern = (features / max_intensity) * 100.0
            else:
                normalized_pattern = features
            for noise_idx in range(2):
                final_features = normalized_pattern.copy()
                if peak_shape_idx != 2: 
                    x_axis = np.arange(5, 120, x_step)
                    features_with_bg = add_polynomial_background(final_features, x_axis)
                    final_features = apply_poisson_noise(features_with_bg)
                    # final_features += np.random.uniform(0, 5, size=final_features.shape)
                
                labels7 = int(crystal_system) - 1
                labels230 = int(space_group) - 1
                
                if save_path:
                    base_name = f"ID_{row.key_value_pairs.get('mpid')[:-4]}"
                    name = os.path.join(save_path, f'{base_name}_{peak_shape_idx}_{noise_idx}.npy')
                    os.makedirs(os.path.dirname(name), exist_ok=True)
                    np.save(name, {'features': final_features, 'labels7': labels7, 'labels230': labels230})
                
                XRDs.append((final_features, labels7, labels230))
                
                if peak_shape_idx == 2:
                    break
        
        if not save_path:
            return XRDs
        return f"Success: ID {row.id}"
        
    except Exception as e:
        return f"Failed: ID {row.id} with error: {e}"


def star(args):
    return process_atoms_row(**args)

def generate_from_ase_db(db_path, save_dir, apply_preferred_orientation=False):
    if not os.path.exists(db_path):
        print(f"can not find '{db_path}' ")
        return

    os.makedirs(save_dir, exist_ok=True)
    
    db = ase.db.connect(db_path)
    
    rows_to_process = []
    for row in tqdm(db.select()):
        check_file = os.path.join(save_dir, f'ID_{row.id}_3_1.npy')
        if not os.path.exists(check_file):
            rows_to_process.append(row)

    if not rows_to_process:
        print("done")
        return

    num_workers = min(os.cpu_count(), 16) 
    
    with mp.Pool(num_workers) as pool:
        args = [dict(row=row, save_path=save_dir, apply_preferred_orientation=apply_preferred_orientation) for row in rows_to_process]
        
        desc = f'from {os.path.basename(db_path)} generate ({len(rows_to_process)}, use {num_workers} cores)'
        
        results = list(tqdm(pool.imap_unordered(star, args), total=len(args), desc=desc))
    
    success_count = sum(1 for r in results if isinstance(r, str) and r.startswith("Success"))
    failed_count = len(results) - success_count
    print(f"\nsuccess: {success_count}, fail: {failed_count}")
    
    if failed_count > 0:
        print("fail example:")
        fail_examples = [r for r in results if isinstance(r, str) and r.startswith("Failed")][:5]
        for example in fail_examples:
            print(f"- {example}")

if __name__ == '__main__':

    ASE_DATABASE_FILE = '../../CrystDB/MP_selected.db' 
    
    OUTPUT_SAVE_DIR = f'{os.path.dirname(__file__)}/Generated/XRDs_from_ASE' 

    generate_from_ase_db(db_path=ASE_DATABASE_FILE, save_dir=OUTPUT_SAVE_DIR, apply_preferred_orientation=True)