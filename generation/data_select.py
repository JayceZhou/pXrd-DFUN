import os
import numpy as np
from ase.db import connect
from ase.spacegroup import get_spacegroup
from tqdm import tqdm
import pickle
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


SOURCE_DB_PATH = '../../../CrystDB/MP.db' 
DEST_DB_PATH = '../../../CrystDB/MP_selected.db'

MY_API_KEY = 'xxxxxxxx'  # Replace with your actual API key if needed

SYMPREC = 1e-5 


def filter_database(source_path, dest_path, symprec):
    if not os.path.exists(source_path):
        print(f"can not find '{source_path}' ")
        return

    source_db = connect(source_path)
    if os.path.exists(dest_path):
        print(f"overlap '{dest_path}'")
        os.remove(dest_path)
    dest_db = connect(dest_path) 

    total_processed = 0
    kept_count = 0
    rejected_discrepancy = 0
    rejected_duplicate = 0
    rejected_spglib_error = 0
    with open('../api_results_cache.pkl', 'rb') as file:
        data = pickle.load(file)
    for row in tqdm(source_db.select(), total=len(source_db)):
        total_processed += 1
        
        try:
            atoms = row.toatoms()

            id = row.key_value_pairs.get('mpid')[:-4]
            sg_number = data[id]['number']
            spg_info = get_spacegroup(atoms, symprec=symprec)
            found_spg = spg_info.no 

            if int(sg_number) != found_spg:
                rejected_discrepancy += 1
                continue 

            dest_db.write(atoms, key_value_pairs=row.key_value_pairs, data=row.get('data'))
            kept_count += 1

        except Exception as e:
            print(e)
            rejected_spglib_error += 1
            continue

    print("-" * 30)
    print(f"total: {total_processed}")
    print(f"success: {kept_count}")
    print("-" * 30)
    print(f"file_path: '{dest_path}'")


if __name__ == '__main__':
    filter_database(SOURCE_DB_PATH, DEST_DB_PATH, SYMPREC)