import os
import numpy as np
from ase.db import connect
from tqdm import tqdm


SOURCE_DB_PATH = '../../CrystDB/MP.db' 
MPID_KEY = 'mpid'
OUTPUT_NPZ_PATH = 'mpids.npy'


def extract_all_mpids(source_path, mpid_key, output_path):
    if not os.path.exists(source_path):
        print(f"can not find '{source_path}' ")
        return

    source_db = connect(source_path)
    
    mpid_list = []
    
    for row in tqdm(source_db.select(), total=len(source_db)):
        try:
            mpid = row.key_value_pairs.get(mpid_key)
            if mpid:
                if isinstance(mpid, str) and mpid.endswith('.cif'):
                    mpid = mpid[:-4]
                
                mpid_list.append(mpid)

        except Exception as e:
            print(f" ID {row.id}, erorr: {e}")
            continue
            
    print("-" * 30)
    print(f"find {len(mpid_list)}  MP ID ")
    
    mpids_array = np.array(mpid_list)
    np.save(output_path, mpids_array)
    
    print(f"saved: '{output_path}'")

if __name__ == '__main__':
    extract_all_mpids(SOURCE_DB_PATH, MPID_KEY, OUTPUT_NPZ_PATH)
