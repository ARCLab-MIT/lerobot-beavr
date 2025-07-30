import pickle
import os
import hashlib
import json
import numpy as np

PKL_PATH = os.path.join(os.path.dirname(__file__), 'test_episodes_random2.pkl')
OUT_PATH = os.path.join(os.path.dirname(__file__), 'unique_episodes_random2.pkl')

def canonical_hash(traj):
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
    print(f"Loaded {len(episodes)} episodes from {PKL_PATH}")
    hash_to_index = {}
    for idx, traj in enumerate(episodes):
        h = canonical_hash(traj)
        if h not in hash_to_index:
            hash_to_index[h] = idx
    unique_indices = sorted(hash_to_index.values())
    unique_episodes = [episodes[i] for i in unique_indices]
    with open(OUT_PATH, 'wb') as f:
        pickle.dump(unique_episodes, f)
    print(f"Saved {len(unique_episodes)} unique episodes to {OUT_PATH}")
    print(f"Indices of unique episodes: {unique_indices}")

if __name__ == "__main__":
    main() 