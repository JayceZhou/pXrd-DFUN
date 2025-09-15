from pymatgen.ext.matproj import MPRester
from tqdm import tqdm
import pickle
import time
import os
import numpy as np

MY_API_KEY = 'YOUR_API_KEY' # replace with your actual API key
MPIDS_FILE_PATH = 'mpids.npy' 
API_CACHE_PATH = 'api_results_cache.pkl'
API_BATCH_SIZE = 100


def batch_query_api(mpids_path, cache_path, api_key, batch_size):

    if not os.path.exists(mpids_path):
        print(f"can not find '{mpids_path}' ")
        return

    mpids = np.load(mpids_path, allow_pickle=True).tolist()
    print(f"load {len(mpids)} MP ID")
    
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            api_results_cache = pickle.load(f)
        
        cached_ids = set(api_results_cache.keys())
        ids_to_query = [mpid for mpid in mpids if mpid not in cached_ids]
    else:
        api_results_cache = {}
        ids_to_query = mpids

    if not ids_to_query:
        return

    print("\nSearch Materials Project API...")
    if not api_key or api_key == "YOUR_API_KEY":
        print("no api key provided, can not query MP API")
        return

    with MPRester(api_key) as mpr:
        num_batches = (len(ids_to_query) + batch_size - 1) // batch_size
        
        with tqdm(total=len(ids_to_query), desc="Querying MP API") as progress_bar:
            for i in range(num_batches):
                start_index = i * batch_size
                end_index = start_index + batch_size
                batch_ids = ids_to_query[start_index:end_index]
                
                if not batch_ids:
                    continue
                try:
                    docs = mpr.summary_search(
                        material_ids=",".join(batch_ids), 
                    )
                    
                    for doc in docs:
                        if doc and doc['symmetry']:
                            api_results_cache[doc['material_id']] = {
                                "number": doc['symmetry']['number'],
                                "crystal_system": doc['symmetry']['crystal_system']
                            }
                    
                    progress_bar.update(len(batch_ids))

                except Exception as e:
                    print(f"\nbatch {i+1} erorr: {e}")
                    time.sleep(5)
    
    with open(cache_path, 'wb') as f:
        pickle.dump(api_results_cache, f)
    print(f"saved: '{cache_path}'")


if __name__ == '__main__':
    batch_query_api(MPIDS_FILE_PATH, API_CACHE_PATH, MY_API_KEY, API_BATCH_SIZE)
