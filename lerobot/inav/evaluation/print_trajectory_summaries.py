import pickle
import os
import numpy as np
import hashlib
import json

T0_TOL = 1e-6  # Tolerance for detecting t=0

PKL_PATH = os.path.join(os.path.dirname(__file__), 'test_episodes.pkl')

def canonical_hash(traj):
    # Convert trajectory dict to a canonical JSON string (sorted keys, lists to tuples)
    def convert(v):
        if isinstance(v, list):
            return tuple(convert(x) for x in v)
        elif isinstance(v, dict):
            return {k: convert(v[k]) for k in sorted(v)}
        elif isinstance(v, np.ndarray):
            return tuple(v.tolist())
        else:
            return v
    canon = convert(traj)
    canon_str = json.dumps(canon, sort_keys=True)
    return hashlib.md5(canon_str.encode('utf-8')).hexdigest()

def main():
    with open(PKL_PATH, 'rb') as f:
        episodes = pickle.load(f)
    print(f"Type of loaded object: {type(episodes)}")
    print(f"Number of trajectories (episodes): {len(episodes)}")
    if len(episodes) > 0:
        print(f"Type of first element: {type(episodes[0])}")
        if isinstance(episodes[0], dict):
            print(f"Keys in first element: {list(episodes[0].keys())}")
            for k in episodes[0]:
                try:
                    print(f"  {k}: len={len(episodes[0][k])}")
                except Exception:
                    print(f"  {k}: type={type(episodes[0][k])}")
    print()

    # Identify unique episodes by hash
    hash_to_index = {}
    for idx, traj in enumerate(episodes):
        h = canonical_hash(traj)
        if h not in hash_to_index:
            hash_to_index[h] = idx
    unique_indices = sorted(hash_to_index.values())
    print(f"Number of unique episodes: {len(unique_indices)}")
    print(f"Indices of first occurrence of each unique episode (first 20): {unique_indices[:20]}{' ...' if len(unique_indices) > 20 else ''}")

if __name__ == "__main__":
    main() 