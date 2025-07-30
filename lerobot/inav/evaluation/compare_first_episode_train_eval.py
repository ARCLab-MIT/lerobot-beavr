import pickle
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

EVAL_PKL_PATH = os.path.join(os.path.dirname(__file__), 'unique_episodes_random.pkl')
TRAIN_PARQUET_PATH = os.path.join(os.path.dirname(__file__), '../../../datasets/iss_docking_images/data/chunk-000/episode_000000.parquet')

# State and action field names for training parquet
STATE_NAMES = [
    'x', 'y', 'z', 'vx', 'vy', 'vz', 'q0', 'q1', 'q2', 'q3', 'w1', 'w2', 'w3'
]
ACTION_NAMES = [
    'Tx', 'Ty', 'Tz', 'Lx', 'Ly', 'Lz'
]
COLORS = ['red', 'blue', 'green']


def print_stats(arr, name):
    arr = np.array(arr)
    print(f"{name}: shape={arr.shape}, mean={arr.mean(axis=0)}, std={arr.std(axis=0)}")

def compute_action_smoothness_il(eval_pkl_path, verbose=True, plot_hist=True):
    """
    Compute and display the mean and std of the L2 norm of first-order finite differences (smoothness metric)
    for all IL (evaluation) episodes in the given pickle file.
    """
    import matplotlib.pyplot as plt
    with open(eval_pkl_path, 'rb') as f:
        episodes = pickle.load(f)
    all_deltas = []
    for ep in episodes:
        # Actions: T (thrust) and L (torque), both (N-1, 3)
        T = np.array(ep['T'])  # (N-1, 3)
        L = np.array(ep['L'])  # (N-1, 3)
        actions = np.concatenate([T, L], axis=1)  # (N-1, 6)
        if actions.shape[0] < 2:
            continue
        delta = actions[1:] - actions[:-1]  # (N-2, 6)
        smoothness_metric = np.linalg.norm(delta, axis=1)  # (N-2,)
        all_deltas.append(smoothness_metric)
    if not all_deltas:
        print("No valid IL episodes found for smoothness metric.")
        return
    all_deltas = np.concatenate(all_deltas)
    if verbose:
        print(f"IL (eval) action smoothness: Mean Δa = {all_deltas.mean():.6f}, Std Δa = {all_deltas.std():.6f}, N={len(all_deltas)}")
    if plot_hist:
        plt.figure(figsize=(7,4))
        plt.hist(all_deltas, bins=50, alpha=0.7, color='blue')
        plt.xlabel('L2 norm of Δa', fontsize=36, labelpad=15)
        plt.ylabel('Count', fontsize=36, labelpad=15)
        plt.tick_params(axis='both', which='major', labelsize=36)
        plt.tight_layout()
        plt.show()


def compute_action_smoothness_rl(rl_dir, verbose=True, plot_hist=True, max_episodes=None):
    """
    Compute and display the mean and std of the L2 norm of first-order finite differences (smoothness metric)
    for all RL (train) episodes in the given directory of parquet files.
    """
    import glob
    import pandas as pd
    import matplotlib.pyplot as plt
    parquet_files = sorted(glob.glob(os.path.join(rl_dir, 'episode_*.parquet')))
    if max_episodes is not None:
        parquet_files = parquet_files[:max_episodes]
    all_deltas = []
    for pf in parquet_files:
        try:
            df = pd.read_parquet(pf)
            actions = np.stack(df['action'].to_numpy())  # (N, 6)
            if actions.shape[0] < 2:
                continue
            delta = actions[1:] - actions[:-1]  # (N-1, 6)
            smoothness_metric = np.linalg.norm(delta, axis=1)  # (N-1,)
            all_deltas.append(smoothness_metric)
        except Exception as e:
            if verbose:
                print(f"Failed to process {pf}: {e}")
    if not all_deltas:
        print("No valid RL episodes found for smoothness metric.")
        return
    all_deltas = np.concatenate(all_deltas)
    if verbose:
        print(f"RL (train) action smoothness: Mean Δa = {all_deltas.mean():.6f}, Std Δa = {all_deltas.std():.6f}, N={len(all_deltas)}")
    if plot_hist:
        plt.figure(figsize=(7,4))
        plt.hist(all_deltas, bins=50, alpha=0.7, color='orange')
        plt.xlabel('L2 norm of Δa', fontsize=36, labelpad=15)
        plt.ylabel('Count', fontsize=36, labelpad=15)
        plt.tick_params(axis='both', which='major', labelsize=36)
        plt.tight_layout()
        plt.show()


def main():
    # Load evaluation episode
    with open(EVAL_PKL_PATH, 'rb') as f:
        eval_episodes = pickle.load(f)
    # Find the best episode: lowest distance at final state (norm of last x[:3])
    best_idx = np.argmin([np.linalg.norm(np.array(ep['x'])[-1][:3]) for ep in eval_episodes])
    eval_ep = eval_episodes[best_idx]
    print(f"Selected best episode index: {best_idx}")
    eval_x = np.array(eval_ep['x'])  # (N, D)
    eval_T = np.array(eval_ep['T'])  # (N-1, 3)
    eval_L = np.array(eval_ep['L'])  # (N-1, 3)
    eval_time = np.array(eval_ep['time'])

    # Load training episode
    train_df = pd.read_parquet(TRAIN_PARQUET_PATH)
    train_state = np.stack(train_df['observation.state'].to_numpy())  # (N, 13)
    train_action = np.stack(train_df['action'].to_numpy())  # (N, 6)
    train_time = np.arange(len(train_state))  # If no time, use index
    if 'timestamp' in train_df:
        train_time = train_df['timestamp'].to_numpy().flatten()

    # --- Plot position (x, y, z) over time ---
    plt.figure(figsize=(10, 6))
    for i, label in enumerate(['x', 'y', 'z']):
        plt.plot(eval_time, eval_x[:, i], color=COLORS[i], linestyle='-', label=f'ACT {label}')
        plt.plot(train_time, train_state[:, i], color=COLORS[i], linestyle='--', label=f'RL {label}')
    plt.xlabel('Time step (s)', fontsize=24)
    plt.ylabel('Position (m)', fontsize=24)
    # Add title with larger font size
    plt.legend(fontsize=18)
    plt.tight_layout()
    plt.tick_params(axis='both', which='major', labelsize=24)
    plt.grid(True)  # Add grid lines
    plt.show()

    # --- Plot actions: Thrust (Tx, Ty, Tz) and Torque (Lx, Ly, Lz) ---
    fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    # Thrust
    for i, label in enumerate(['x', 'y', 'z']):
        axs[0].plot(np.arange(len(eval_T)), eval_T[:, i], color=COLORS[i], linestyle='-', label=f'ACT {label}')
        axs[0].plot(np.arange(len(train_action)), train_action[:, i], color=COLORS[i], linestyle='--', label=f'RL {label}')
    axs[0].set_ylabel('Thrust (N)', fontsize=24)
    axs[0].grid(True)  # Add grid lines
    # Torque
    for i, label in enumerate(['x', 'y', 'z']):
        axs[1].plot(np.arange(len(eval_L)), eval_L[:, i], color=COLORS[i], linestyle='-', label=f'ACT {label}')
        axs[1].plot(np.arange(len(train_action)), train_action[:, i+3], color=COLORS[i], linestyle='--', label=f'RL {label}')
    axs[1].set_xlabel('Time step (s)', fontsize=24)
    axs[1].set_ylabel('Torque (Nm)', fontsize=24)
    axs[1].grid(True)  # Add grid lines
    # Set tick label size for both subplots
    for ax in axs:
        ax.tick_params(axis='both', which='major', labelsize=24)
    # Create a single legend with concise labels
    handles1, labels1 = axs[1].get_legend_handles_labels()
    axs[1].legend(handles1, labels1, fontsize=18)
    plt.tight_layout()
    plt.show()

    # --- Smoothness metric for all episodes ---
    print("\n=== Action Smoothness Metric (L2 norm of Δa) ===")
    compute_action_smoothness_il(EVAL_PKL_PATH, verbose=True, plot_hist=True)
    RL_DATA_DIR = os.path.join(os.path.dirname(__file__), '../../../datasets/iss_docking_images/data/chunk-000')
    compute_action_smoothness_rl(RL_DATA_DIR, verbose=True, plot_hist=True, max_episodes=None)  # Set max_episodes=None for all

    # --- Compare initial x positions for each episode pair ---
    import glob
    rl_dir = os.path.join(os.path.dirname(__file__), '../../../datasets/iss_docking_images/data/chunk-000')
    parquet_files = sorted(glob.glob(os.path.join(rl_dir, 'episode_*.parquet')))
    n_pairs = min(len(parquet_files), len(eval_episodes))
    x_diffs = []
    for i in range(n_pairs):
        # RL episode
        df = pd.read_parquet(parquet_files[i])
        rl_state = np.stack(df['observation.state'].to_numpy())
        rl_x0 = rl_state[0, 0]  # initial x
        # Eval episode
        eval_x0 = np.array(eval_episodes[i]['x'])[0, 0]  # initial x
        x_diffs.append(eval_x0 - rl_x0)
    x_diffs = np.array(x_diffs)
    print("\n=== Initial X Position Difference (Eval - RL) for Each Episode ===")
    print(f"Compared {n_pairs} episode pairs.")
    print(f"Mean difference: {x_diffs.mean():.6f}")
    print(f"Range: {x_diffs.min():.6f} to {x_diffs.max():.6f}")
    print(f"Average absolute difference: {np.abs(x_diffs).mean():.6f}")


if __name__ == "__main__":
    # print first 3 episodes of pkl file
    with open(EVAL_PKL_PATH, 'rb') as f:
        eval_episodes = pickle.load(f)
    for i in range(3):
        print(eval_episodes[i])
        print() 
    main() 