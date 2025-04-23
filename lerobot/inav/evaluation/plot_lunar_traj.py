import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import ast
from matplotlib.gridspec import GridSpec
import pyarrow.parquet as pq

def plot_lunar_trajectory(csv_path):
    """
    Plot the thrusts and torques from a lunar trajectory inference CSV file.
    
    Args:
        csv_path: Path to the inference CSV file
    """
    # Load the data
    df = pd.read_csv(csv_path)
    
    # Convert string representation of lists to actual lists
    df['action_list'] = df['action'].apply(ast.literal_eval)
    
    # Extract thrusts (first 3 elements) and torques (last 3 elements)
    thrusts = np.array([action[:3] for action in df['action_list']])
    torques = np.array([action[3:] for action in df['action_list']])
    
    # Create figure with GridSpec for better layout control
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(3, 2, figure=fig)
    
    # Plot thrusts
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(thrusts[:, 0], label='Thrust X (N)', linewidth=2)
    ax1.plot(thrusts[:, 1], label='Thrust Y (N)', linewidth=2)
    ax1.plot(thrusts[:, 2], label='Thrust Z (N)', linewidth=2)
    ax1.axhline(0, color='lightgray', linewidth=0.8, linestyle='--')
    ax1.grid(False)
    ax1.tick_params(axis='both', which='major', labelsize=14)
    ax1.set_title('Thrusts over Time', fontsize=22)
    ax1.set_xlabel('Frame Index')
    ax1.set_ylabel('Thrust (N)')
    
    # Plot torques
    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(torques[:, 0], label='Torque Roll (N·m)', linewidth=2)
    ax2.plot(torques[:, 1], label='Torque Pitch (N·m)', linewidth=2)
    ax2.plot(torques[:, 2], label='Torque Yaw (N·m)', linewidth=2)
    ax2.axhline(0, color='lightgray', linewidth=0.8, linestyle='--')
    ax2.grid(False)
    ax2.tick_params(axis='both', which='major', labelsize=14)
    ax2.set_title('Torques over Time', fontsize=22)
    ax2.set_xlabel('Frame Index')
    ax2.set_ylabel('Torque (N·m)')
    
    # Plot thrust magnitude
    ax3 = fig.add_subplot(gs[2, 0])
    thrust_magnitude = np.sqrt(np.sum(thrusts**2, axis=1))
    ax3.plot(thrust_magnitude, color='purple', linewidth=2)
    ax3.axhline(0, color='lightgray', linewidth=0.8, linestyle='--')
    ax3.grid(False)
    ax3.tick_params(axis='both', which='major', labelsize=14)
    ax3.set_title('Total Thrust Magnitude', fontsize=22)
    ax3.set_xlabel('Frame Index')
    ax3.set_ylabel('Magnitude (N)')
    ax3.set_ylim(0, 10000)  # Set y-axis range for thrust magnitude
    
    # Plot torque magnitude
    ax4 = fig.add_subplot(gs[2, 1])
    torque_magnitude = np.sqrt(np.sum(torques**2, axis=1))
    ax4.plot(torque_magnitude, color='darkgreen', linewidth=2)
    ax4.axhline(0, color='lightgray', linewidth=0.8, linestyle='--')
    ax4.grid(False)
    ax4.tick_params(axis='both', which='major', labelsize=14)
    ax4.set_title('Total Torque Magnitude', fontsize=22)
    ax4.set_xlabel('Frame Index')
    ax4.set_ylabel('Magnitude (N·m)')
    
    # Add overall title and adjust layout
    # plt.suptitle('ACT', fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save the figure
    output_dir = os.path.dirname(csv_path)
    episode_name = os.path.basename(csv_path).split('_inference')[0]
    output_path = os.path.join(output_dir, f"{episode_name}_thrust_torque_plot.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    return fig

def plot_lunar_trajectory_separate(csv_path):
    """
    Plot the thrusts and torques from a lunar trajectory inference CSV file in separate plots.
    
    Args:
        csv_path: Path to the inference CSV file
    """
    # Load the data
    df = pd.read_csv(csv_path)
    
    # Convert string representation of lists to actual lists
    df['action_list'] = df['action'].apply(ast.literal_eval)
    
    # Extract thrusts (first 3 elements) and torques (last 3 elements)
    thrusts = np.array([action[:3] for action in df['action_list']])
    torques = np.array([action[3:] for action in df['action_list']])
    
    # Plot thrusts
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(thrusts[:, 0], label='Thrust X (N)', linewidth=2)
    ax1.plot(thrusts[:, 1], label='Thrust Y (N)', linewidth=2)
    ax1.plot(thrusts[:, 2], label='Thrust Z (N)', linewidth=2)
    ax1.axhline(0, color='lightgray', linewidth=0.8, linestyle='--')
    ax1.grid(False)
    ax1.tick_params(axis='both', which='major', labelsize=18)
    ax1.set_xlabel('Frame Index', fontsize=24)
    ax1.set_ylabel('Thrust (N)', fontsize=24)
    ax1.set_ylim(-2000, 8000)
    ax1.legend(fontsize=18)
    output_path1 = os.path.join(os.path.dirname(csv_path), "thrusts_plot.pdf")
    plt.savefig(output_path1, format='pdf', bbox_inches='tight')
    print(f"Thrusts plot saved to: {output_path1}")
    
    # Plot torques
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(torques[:, 0], label='Torque Roll (N·m)', linewidth=2)
    ax2.plot(torques[:, 1], label='Torque Pitch (N·m)', linewidth=2)
    ax2.plot(torques[:, 2], label='Torque Yaw (N·m)', linewidth=2)
    ax2.axhline(0, color='lightgray', linewidth=0.8, linestyle='--')
    ax2.grid(False)
    ax2.tick_params(axis='both', which='major', labelsize=18)
    ax2.set_xlabel('Frame Index', fontsize=24)
    ax2.set_ylabel('Torque (N·m)', fontsize=24)
    ax2.legend(fontsize=18)
    output_path2 = os.path.join(os.path.dirname(csv_path), "torques_plot.pdf")
    plt.savefig(output_path2, format='pdf', bbox_inches='tight')
    print(f"Torques plot saved to: {output_path2}")
    
    # Plot thrust magnitude
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    thrust_magnitude = np.sqrt(np.sum(thrusts**2, axis=1))
    ax3.plot(thrust_magnitude, color='purple', linewidth=2)
    ax3.axhline(0, color='lightgray', linewidth=0.8, linestyle='--')
    ax3.grid(False)
    ax3.tick_params(axis='both', which='major', labelsize=18)
    ax3.set_xlabel('Frame Index', fontsize=24)
    ax3.set_ylabel('Magnitude (N)', fontsize=24)
    ax3.set_ylim(0, 10000)
    output_path3 = os.path.join(os.path.dirname(csv_path), "thrust_magnitude_plot.pdf")
    plt.savefig(output_path3, format='pdf', bbox_inches='tight')
    print(f"Thrust magnitude plot saved to: {output_path3}")
    
    # Plot torque magnitude
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    torque_magnitude = np.sqrt(np.sum(torques**2, axis=1))
    ax4.plot(torque_magnitude, color='darkgreen', linewidth=2)
    ax4.axhline(0, color='lightgray', linewidth=0.8, linestyle='--')
    ax4.grid(False)
    ax4.tick_params(axis='both', which='major', labelsize=18)
    ax4.set_xlabel('Frame Index', fontsize=24)
    ax4.set_ylabel('Magnitude (N·m)', fontsize=24)
    output_path4 = os.path.join(os.path.dirname(csv_path), "torque_magnitude_plot.pdf")
    plt.savefig(output_path4, format='pdf', bbox_inches='tight')
    print(f"Torque magnitude plot saved to: {output_path4}")
    
    return fig1, fig2, fig3, fig4

def plot_parquet_actions(parquet_path):
    """
    Plot the thrusts and torques from actions in a Parquet file.
    
    Args:
        parquet_path: Path to the Parquet file
    """
    # Load the data
    table = pq.read_table(parquet_path)
    df = table.to_pandas()
    
    # Extract actions
    actions = np.stack(df['action'].values)
    
    # Extract thrusts (first 3 elements) and torques (last 3 elements)
    thrusts = actions[:, :3]
    torques = actions[:, 3:]
    
    # Create figure with GridSpec for better layout control
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(3, 2, figure=fig)
    
    # Plot thrusts
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(thrusts[:, 0], label='Thrust X (N)', linewidth=2)
    ax1.plot(thrusts[:, 1], label='Thrust Y (N)', linewidth=2)
    ax1.plot(thrusts[:, 2], label='Thrust Z (N)', linewidth=2)
    ax1.axhline(0, color='lightgray', linewidth=0.8, linestyle='--')
    ax1.grid(False)
    ax1.tick_params(axis='both', which='major', labelsize=14)
    ax1.set_title('Thrusts over Time', fontsize=22)
    ax1.set_xlabel('Frame Index')
    ax1.set_ylabel('Thrust (N)')
    ax1.set_ylim(-2000, 8000)  # Set y-axis range for thrusts
    ax1.legend()
    
    # Plot torques
    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(torques[:, 0], label='Torque Roll (N·m)', linewidth=2)
    ax2.plot(torques[:, 1], label='Torque Pitch (N·m)', linewidth=2)
    ax2.plot(torques[:, 2], label='Torque Yaw (N·m)', linewidth=2)
    ax2.axhline(0, color='lightgray', linewidth=0.8, linestyle='--')
    ax2.grid(False)
    ax2.tick_params(axis='both', which='major', labelsize=14)
    ax2.set_title('Torques over Time', fontsize=22)
    ax2.set_xlabel('Frame Index')
    ax2.set_ylabel('Torque (N·m)')
    ax2.legend()
    
    # Plot thrust magnitude
    ax3 = fig.add_subplot(gs[2, 0])
    thrust_magnitude = np.sqrt(np.sum(thrusts**2, axis=1))
    ax3.plot(thrust_magnitude, color='purple', linewidth=2)
    ax3.axhline(0, color='lightgray', linewidth=0.8, linestyle='--')
    ax3.grid(False)
    ax3.tick_params(axis='both', which='major', labelsize=14)
    ax3.set_title('Total Thrust Magnitude', fontsize=22)
    ax3.set_xlabel('Frame Index')
    ax3.set_ylabel('Magnitude (N)')
    ax3.set_ylim(0, 10000)  # Set y-axis range for thrust magnitude
    
    # Plot torque magnitude
    ax4 = fig.add_subplot(gs[2, 1])
    torque_magnitude = np.sqrt(np.sum(torques**2, axis=1))
    ax4.plot(torque_magnitude, color='darkgreen', linewidth=2)
    ax4.axhline(0, color='lightgray', linewidth=0.8, linestyle='--')
    ax4.grid(False)
    ax4.tick_params(axis='both', which='major', labelsize=14)
    ax4.set_title('Total Torque Magnitude', fontsize=22)
    ax4.set_xlabel('Frame Index')
    ax4.set_ylabel('Magnitude (N·m)')
    
    # Add overall title and adjust layout
    plt.suptitle('Meta-RL', fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save the figure
    output_dir = os.path.dirname(parquet_path)
    episode_name = os.path.basename(parquet_path).split('.')[0]
    output_path = os.path.join(output_dir, f"{episode_name}_actions_thrust_torque_plot.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Actions plot saved to: {output_path}")
    
    return fig

def plot_parquet_actions_separate(parquet_path):
    """
    Plot the thrusts and torques from actions in a Parquet file in separate plots.
    
    Args:
        parquet_path: Path to the Parquet file
    """
    # Load the data
    table = pq.read_table(parquet_path)
    df = table.to_pandas()
    
    # Extract actions
    actions = np.stack(df['action'].values)
    
    # Extract thrusts (first 3 elements) and torques (last 3 elements)
    thrusts = actions[:, :3]
    torques = actions[:, 3:]
    
    # Plot thrusts
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(thrusts[:, 0], label='Thrust X (N)', linewidth=2)
    ax1.plot(thrusts[:, 1], label='Thrust Y (N)', linewidth=2)
    ax1.plot(thrusts[:, 2], label='Thrust Z (N)', linewidth=2)
    ax1.axhline(0, color='lightgray', linewidth=0.8, linestyle='--')
    ax1.grid(False)
    ax1.tick_params(axis='both', which='major', labelsize=18)
    ax1.set_xlabel('Frame Index', fontsize=24)
    ax1.set_ylabel('Thrust (N)', fontsize=24)
    ax1.set_ylim(-2000, 8000)
    ax1.legend(fontsize=18)
    output_dir = os.path.dirname(parquet_path)
    episode_name = os.path.basename(parquet_path).split('.')[0]
    output_path1 = os.path.join(output_dir, f"metarl_{episode_name}_thrusts_plot.pdf")
    plt.savefig(output_path1, format='pdf', bbox_inches='tight')
    print(f"Parquet thrusts plot saved to: {output_path1}")
    
    # Plot torques
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(torques[:, 0], label='Torque Roll (N·m)', linewidth=2)
    ax2.plot(torques[:, 1], label='Torque Pitch (N·m)', linewidth=2)
    ax2.plot(torques[:, 2], label='Torque Yaw (N·m)', linewidth=2)
    ax2.axhline(0, color='lightgray', linewidth=0.8, linestyle='--')
    ax2.grid(False)
    ax2.tick_params(axis='both', which='major', labelsize=18)
    ax2.set_xlabel('Frame Index', fontsize=24)
    ax2.set_ylabel('Torque (N·m)', fontsize=24)
    ax2.legend(fontsize=18)
    output_path2 = os.path.join(output_dir, f"metarl_{episode_name}_torques_plot.pdf")
    plt.savefig(output_path2, format='pdf', bbox_inches='tight')
    print(f"Parquet torques plot saved to: {output_path2}")
    
    # Plot thrust magnitude
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    thrust_magnitude = np.sqrt(np.sum(thrusts**2, axis=1))
    ax3.plot(thrust_magnitude, color='purple', linewidth=2)
    ax3.axhline(0, color='lightgray', linewidth=0.8, linestyle='--')
    ax3.grid(False)
    ax3.tick_params(axis='both', which='major', labelsize=18)
    ax3.set_xlabel('Frame Index', fontsize=24)
    ax3.set_ylabel('Magnitude (N)', fontsize=24)
    ax3.set_ylim(0, 10000)
    output_path3 = os.path.join(output_dir, f"metarl_{episode_name}_thrust_magnitude_plot.pdf")
    plt.savefig(output_path3, format='pdf', bbox_inches='tight')
    print(f"Parquet thrust magnitude plot saved to: {output_path3}")
    
    # Plot torque magnitude
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    torque_magnitude = np.sqrt(np.sum(torques**2, axis=1))
    ax4.plot(torque_magnitude, color='darkgreen', linewidth=2)
    ax4.axhline(0, color='lightgray', linewidth=0.8, linestyle='--')
    ax4.grid(False)
    ax4.tick_params(axis='both', which='major', labelsize=18)
    ax4.set_xlabel('Frame Index', fontsize=24)
    ax4.set_ylabel('Magnitude (N·m)', fontsize=24)
    output_path4 = os.path.join(output_dir, f"metarl_{episode_name}_torque_magnitude_plot.pdf")
    plt.savefig(output_path4, format='pdf', bbox_inches='tight')
    print(f"Parquet torque magnitude plot saved to: {output_path4}")
    
    return fig1, fig2, fig3, fig4

def plot_combined_comparison(csv_path, parquet_path):
    """
    Plot the thrusts and torques from both lunar trajectory inference CSV and Parquet file
    in 4 separate plots for direct comparison.
    
    Args:
        csv_path: Path to the inference CSV file
        parquet_path: Path to the Parquet file
    
    Returns:
        tuple: The four figure objects created
    """
    # Load CSV data
    df_csv = pd.read_csv(csv_path)
    df_csv['action_list'] = df_csv['action'].apply(ast.literal_eval)
    csv_thrusts = np.array([action[:3] for action in df_csv['action_list']])
    csv_torques = np.array([action[3:] for action in df_csv['action_list']])
    
    # Load Parquet data
    table = pq.read_table(parquet_path)
    df_parquet = table.to_pandas()
    actions = np.stack(df_parquet['action'].values)
    parquet_thrusts = actions[:, :3]
    parquet_torques = actions[:, 3:]
    
    # Calculate magnitudes
    csv_thrust_magnitude = np.sqrt(np.sum(csv_thrusts**2, axis=1))
    csv_torque_magnitude = np.sqrt(np.sum(csv_torques**2, axis=1))
    parquet_thrust_magnitude = np.sqrt(np.sum(parquet_thrusts**2, axis=1))
    parquet_torque_magnitude = np.sqrt(np.sum(parquet_torques**2, axis=1))
    
    # Get episode names for the legend
    csv_episode = os.path.basename(csv_path).split('_inference')[0]
    parquet_episode = os.path.basename(parquet_path).split('.')[0]
    
    # Create output directory
    output_dir = os.path.dirname(csv_path)
    
    # 1. Plot thrusts comparison
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    # CSV thrusts with solid lines
    ax1.plot(csv_thrusts[:, 0], label=f'ACT X (N)', linewidth=2, linestyle='-')
    ax1.plot(csv_thrusts[:, 1], label=f'ACT Y (N)', linewidth=2, linestyle='-')
    ax1.plot(csv_thrusts[:, 2], label=f'ACT Z (N)', linewidth=2, linestyle='-')
    # Parquet thrusts with dashed lines
    ax1.plot(parquet_thrusts[:, 0], label=f'Meta-RL X (N)', linewidth=2, linestyle='--')
    ax1.plot(parquet_thrusts[:, 1], label=f'Meta-RL Y (N)', linewidth=2, linestyle='--')
    ax1.plot(parquet_thrusts[:, 2], label=f'Meta-RL Z (N)', linewidth=2, linestyle='--')
    
    ax1.axhline(0, color='lightgray', linewidth=0.8, linestyle='--')
    ax1.grid(False)
    ax1.tick_params(axis='both', which='major', labelsize=18)
    ax1.set_xlabel('Frame Index', fontsize=24)
    ax1.set_ylabel('Thrust (N)', fontsize=24)
    ax1.set_ylim(-2000, 8000)
    ax1.legend(fontsize=18)
    ax1.set_title('Thrust Comparison', fontsize=26)
    
    output_path1 = os.path.join(output_dir, f"comparison_thrusts_plot.pdf")
    plt.savefig(output_path1, format='pdf', bbox_inches='tight')
    print(f"Comparison thrusts plot saved to: {output_path1}")
    
    # 2. Plot torques comparison
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    # CSV torques with solid lines
    ax2.plot(csv_torques[:, 0], label=f'ACT Roll (N·m)', linewidth=2, linestyle='-')
    ax2.plot(csv_torques[:, 1], label=f'ACT Pitch (N·m)', linewidth=2, linestyle='-')
    ax2.plot(csv_torques[:, 2], label=f'ACT Yaw (N·m)', linewidth=2, linestyle='-')
    # Parquet torques with dashed lines
    ax2.plot(parquet_torques[:, 0], label=f'Meta-RL Roll (N·m)', linewidth=2, linestyle='--')
    ax2.plot(parquet_torques[:, 1], label=f'Meta-RL Pitch (N·m)', linewidth=2, linestyle='--')
    ax2.plot(parquet_torques[:, 2], label=f'Meta-RL Yaw (N·m)', linewidth=2, linestyle='--')
    
    ax2.axhline(0, color='lightgray', linewidth=0.8, linestyle='--')
    ax2.grid(False)
    ax2.tick_params(axis='both', which='major', labelsize=18)
    ax2.set_xlabel('Frame Index', fontsize=24)
    ax2.set_ylabel('Torque (N·m)', fontsize=24)
    ax2.legend(fontsize=18)
    ax2.set_title('Torque Comparison', fontsize=26)
    
    output_path2 = os.path.join(output_dir, f"comparison_torques_plot.pdf")
    plt.savefig(output_path2, format='pdf', bbox_inches='tight')
    print(f"Comparison torques plot saved to: {output_path2}")
    
    # 3. Plot thrust magnitude comparison
    fig3, ax3 = plt.subplots(figsize=(12, 8))
    ax3.plot(csv_thrust_magnitude, label='ACT', color='blue', linewidth=2, linestyle='-')
    ax3.plot(parquet_thrust_magnitude, label='Meta-RL', color='red', linewidth=2, linestyle='--')
    ax3.axhline(0, color='lightgray', linewidth=0.8, linestyle='--')
    ax3.grid(False)
    ax3.tick_params(axis='both', which='major', labelsize=18)
    ax3.set_xlabel('Frame Index', fontsize=24)
    ax3.set_ylabel('Magnitude (N)', fontsize=24)
    ax3.set_ylim(0, 10000)
    ax3.legend(fontsize=18)
    ax3.set_title('Thrust Magnitude Comparison', fontsize=26)
    
    output_path3 = os.path.join(output_dir, f"comparison_thrust_magnitude_plot.pdf")
    plt.savefig(output_path3, format='pdf', bbox_inches='tight')
    print(f"Comparison thrust magnitude plot saved to: {output_path3}")
    
    # 4. Plot torque magnitude comparison
    fig4, ax4 = plt.subplots(figsize=(12, 8))
    ax4.plot(csv_torque_magnitude, label='ACT', color='blue', linewidth=2, linestyle='-')
    ax4.plot(parquet_torque_magnitude, label='Meta-RL', color='red', linewidth=2, linestyle='--')
    ax4.axhline(0, color='lightgray', linewidth=0.8, linestyle='--')
    ax4.grid(False)
    ax4.tick_params(axis='both', which='major', labelsize=18)
    ax4.set_xlabel('Frame Index', fontsize=24)
    ax4.set_ylabel('Magnitude (N·m)', fontsize=24)
    ax4.legend(fontsize=18)
    ax4.set_title('Torque Magnitude Comparison', fontsize=26)
    
    output_path4 = os.path.join(output_dir, f"comparison_torque_magnitude_plot.pdf")
    plt.savefig(output_path4, format='pdf', bbox_inches='tight')
    print(f"Comparison torque magnitude plot saved to: {output_path4}")
    
    return fig1, fig2, fig3, fig4

def main():
    # Set up command line argument parsing if needed
    import argparse
    parser = argparse.ArgumentParser(description='Plot lunar trajectory thrusts and torques')
    parser.add_argument('--csv_path', type=str, 
                        default='lerobot/inav/evaluation/inference_results/episode_000000_inference.csv',
                        help='Path to the inference CSV file')
    parser.add_argument('--parquet_path', type=str,
                        default='datasets/moon_lander_lerobot/data/chunk-000/episode_000000.parquet',
                        help='Path to the parquet file')
    parser.add_argument('--mode', type=str, default='separate',
                        choices=['combined', 'separate', 'comparison'],
                        help='Plot mode: combined (all in one figure), separate (individual figures), or comparison')
    args = parser.parse_args()
    
    if args.mode == 'combined':
        # Plot the trajectory in a combined figure
        fig = plot_lunar_trajectory(args.csv_path)
        # Plot the actions from the Parquet file in a combined figure
        fig_parquet = plot_parquet_actions(args.parquet_path)
    elif args.mode == 'separate':
        # Plot the trajectory in separate figures
        figs = plot_lunar_trajectory_separate(args.csv_path)
        # Plot the actions from the Parquet file in separate figures
        figs_parquet = plot_parquet_actions_separate(args.parquet_path)
    elif args.mode == 'comparison':
        # Plot comparison between CSV and Parquet data
        figs_comparison = plot_combined_comparison(args.csv_path, args.parquet_path)
    
    # Show the plot if running interactively
    plt.show()

if __name__ == "__main__":
    main()
