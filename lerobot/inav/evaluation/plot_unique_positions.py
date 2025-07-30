import glob
import os
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PKL_PATH = os.path.join(os.path.dirname(__file__), 'unique_episodes_random2.pkl')

# Config flag to enable/disable RL (parquet) plotting
PLOT_RL = True  # Set to False to disable RL data plotting

# Directory containing RL parquet files
RL_PARQUET_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../datasets/iss_docking_images/data/chunk-000'))


def main():
    with open(PKL_PATH, 'rb') as f:
        episodes = pickle.load(f)
    print(f"Loaded {len(episodes)} unique episodes from {PKL_PATH}")

    sample_indices = random.sample(range(len(episodes)), 100)
    print(f"Plotting episodes: {sample_indices}")

    # --- IL (pickle) plot ---
    fig_il = plt.figure(figsize=(10, 8))
    ax_il = fig_il.add_subplot(111, projection='3d')
    cmap = plt.get_cmap('viridis')
    colors = [cmap(i / len(sample_indices)) for i in range(len(sample_indices))]
    for i, idx in enumerate(sample_indices):
        ep = episodes[idx]
        x = np.array(ep['x'])
        if x.shape[1] < 3:
            print(f"Episode {idx}: 'x' does not have at least 3 columns, skipping.")
            continue
        pos = x[:, :3] * 1000
        ax_il.plot(pos[:, 0], pos[:, 1], pos[:, 2], color=colors[i], alpha=0.7, linewidth=0.5)
        ax_il.scatter(pos[0, 0], pos[0, 1], pos[0, 2], c=[colors[i]], marker='o', s=30)
        ax_il.scatter(pos[-1, 0], pos[-1, 1], pos[-1, 2], c=[colors[i]], marker='^', s=30)
    # Draw a 1 unit radius circle in the x-z plane at y=0 (origin)
    theta = np.linspace(0, 2 * np.pi, 100)
    circle_x = 2 * np.cos(theta)
    circle_y = np.zeros_like(theta)
    circle_z = 2 * np.sin(theta)
    ax_il.plot(circle_x, circle_y, circle_z, color='black', linestyle='--', linewidth=2, label='Target Area')
    ax_il.set_xlabel('X (m)', fontsize=18, labelpad=15)
    ax_il.set_ylabel('Y (m)', fontsize=18, labelpad=15)
    ax_il.set_zlabel('Z (m)', fontsize=18, labelpad=15)
    ax_il.tick_params(axis='both', which='major', labelsize=16)
    ax_il.legend(loc='best', fontsize='large')
    plt.tight_layout()
    # Set axis limits for IL
    all_pos_il = np.concatenate([np.array(episodes[idx]['x'])[:, :3] * 1000 for idx in sample_indices], axis=0)
    def expand(lim, factor=0.1):
        delta = (lim[1] - lim[0]) * factor
        return (lim[0] - delta, lim[1] + delta)
    ax_il.set_xlim(expand((all_pos_il[:, 0].min(), all_pos_il[:, 0].max())))
    ax_il.set_ylim(expand((all_pos_il[:, 1].min(), all_pos_il[:, 1].max())))
    ax_il.set_zlim(expand((all_pos_il[:, 2].min(), all_pos_il[:, 2].max())))
    plt.show()

    # --- RL (parquet) plot ---
    if PLOT_RL:
        parquet_files = sorted(glob.glob(os.path.join(RL_PARQUET_DIR, 'episode_*.parquet')))
        if len(parquet_files) == 0:
            print(f"No RL parquet files found in {RL_PARQUET_DIR}")
        else:
            fig_rl = plt.figure(figsize=(10, 8))
            ax_rl = fig_rl.add_subplot(111, projection='3d')
            rl_sample_indices = random.sample(range(len(parquet_files)), min(100, len(parquet_files)))
            rl_cmap = plt.get_cmap('plasma')
            rl_colors = [rl_cmap(i / len(rl_sample_indices)) for i in range(len(rl_sample_indices))]
            all_pos_rl = []
            for i, idx in enumerate(rl_sample_indices):
                pf = parquet_files[idx]
                try:
                    df = pd.read_parquet(pf)
                    state = np.stack(df['observation.state'].to_numpy())
                    if state.shape[1] < 3:
                        print(f"RL episode {pf}: state does not have at least 3 columns, skipping.")
                        continue
                    pos = state[:, :3]
                    all_pos_rl.append(pos)
                    ax_rl.plot(pos[:, 0], pos[:, 1], pos[:, 2], color=rl_colors[i], alpha=0.7, linewidth=0.5, linestyle=':')
                    ax_rl.scatter(pos[0, 0], pos[0, 1], pos[0, 2], c=[rl_colors[i]], marker='s', s=20)
                    ax_rl.scatter(pos[-1, 0], pos[-1, 1], pos[-1, 2], c=[rl_colors[i]], marker='x', s=20)
                except Exception as e:
                    print(f"Failed to process RL episode {pf}: {e}")
            # Draw a 1 unit radius circle in the x-z plane at y=0 (origin)
            ax_rl.plot(circle_x * 1000, circle_y * 1000, circle_z * 1000, color='black', linestyle='--', linewidth=2, label='Target Area')
            ax_rl.set_xlabel('X (m)', fontsize=18, labelpad=15)
            ax_rl.set_ylabel('Y (m)', fontsize=18, labelpad=15)
            ax_rl.set_zlabel('Z (m)', fontsize=18, labelpad=15)
            ax_rl.tick_params(axis='both', which='major', labelsize=16)
            ax_rl.legend(loc='best', fontsize='large')
            plt.tight_layout()
            # Set axis limits for RL
            if all_pos_rl:
                all_pos_rl = np.concatenate(all_pos_rl, axis=0)
                ax_rl.set_xlim(expand((all_pos_rl[:, 0].min(), all_pos_rl[:, 0].max())))
                ax_rl.set_ylim(expand((all_pos_rl[:, 1].min(), all_pos_rl[:, 1].max())))
                ax_rl.set_zlim(expand((all_pos_rl[:, 2].min(), all_pos_rl[:, 2].max())))
            plt.show()

if __name__ == "__main__":
    main()