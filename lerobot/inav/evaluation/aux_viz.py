import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import pickle, os

PKL_PATH = os.path.join(os.path.dirname(__file__), 'unique_episodes_random2.pkl')
episodes = pickle.load(open(PKL_PATH, 'rb'))

def plot_density(episodes, ax=None, levels=8, cmap='Reds', alpha=0.6, axis1=0, axis2=2):
    """
    Plot a 2D density heatmap for any pair of axes (x, y, z) from trajectory data.
    Args:
        episodes: List of episode dicts with 'x' key containing trajectory points.
        ax: Optional matplotlib axis.
        levels: Number of contour levels.
        cmap: Colormap for the heatmap.
        alpha: Transparency for the heatmap.
        axis1: Index for the first axis (0=x, 1=y, 2=z).
        axis2: Index for the second axis (0=x, 1=y, 2=z).
    """
    # Stack all selected axis points from all selected trajectories
    pts = np.vstack([np.array(ep['x'])[:, [axis1, axis2]] for ep in episodes])
    a1, a2 = pts.T
    a1 = a1 * 1000
    a2 = a2 * 1000
    kde = gaussian_kde(np.vstack([a1, a2]))
    a1i = np.linspace(a1.min(), a1.max(), 200)
    a2i = np.linspace(a2.min(), a2.max(), 200)
    A1, A2 = np.meshgrid(a1i, a2i)
    Y = kde(np.vstack([A1.ravel(), A2.ravel()])).reshape(A1.shape)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))  # a wider figure
    ax.contourf(A1, A2, Y, levels=levels, cmap=cmap, alpha=alpha)

    # optional: faint trajectories
    # ax.scatter(a1, a2, c='k', s=1, alpha=0.1)  # Add points overlay

    # drop equal aspect so it fills the rectangle
    ax.set_aspect('auto')
    axis_names = ['X', 'Y', 'Z']
    ax.set_xlabel(f'{axis_names[axis1]} (m)', fontsize=36)
    ax.set_ylabel(f'{axis_names[axis2]} (m)', fontsize=36)
    ax.tick_params(axis='both', which='major', labelsize=30)
    return ax

def plot_histogram(episodes, bins=50):
    """
    Plot histograms of all trajectory waypoints for each axis (x, y, z).
    Args:
        episodes: List of episode dicts with 'x' key containing trajectory points.
        bins: Number of bins for the histogram.
    """
    pts = np.vstack([np.array(ep['x']) for ep in episodes])
    axis_names = ['X', 'Y', 'Z']
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    for i in range(3):
        axs[i].hist(pts[:, i], bins=bins, color='C0', alpha=0.7)
        axs[i].set_xlabel(f'{axis_names[i]} (m)', fontsize=36)
        axs[i].set_ylabel('Count', fontsize=36)
        axs[i].tick_params(axis='both', which='major', labelsize=30)
    plt.tight_layout()
    plt.show()

def plot_density_from_positions(positions, axis1=2, axis2=1, ax=None, levels=8, cmap='Reds', alpha=0.6):
    """
    Plot a 2D density heatmap for any pair of axes from a numpy array of positions.
    Args:
        positions: np.ndarray of shape (N, 3) or (N, D), where columns are [x, y, z, ...].
        axis1: Index for the first axis (default 2=Z).
        axis2: Index for the second axis (default 1=Y).
        ax: Optional matplotlib axis.
        levels, cmap, alpha: Plotting options.
    """
    pts = positions[:, [axis1, axis2]]
    a1, a2 = pts.T
    
    # Print data ranges for debugging
    print(f"Axis {axis1} range: [{a1.min():.3f}, {a1.max():.3f}]")
    print(f"Axis {axis2} range: [{a2.min():.3f}, {a2.max():.3f}]")
    
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(np.vstack([a1, a2]))
    
    # Extend the range slightly beyond min/max for better visualization
    a1_margin = (a1.max() - a1.min()) * 0.1
    a2_margin = (a2.max() - a2.min()) * 0.1
    
    a1i = np.linspace(a1.min() - a1_margin, a1.max() + a1_margin, 200)
    a2i = np.linspace(a2.min() - a2_margin, a2.max() + a2_margin, 200)
    A1, A2 = np.meshgrid(a1i, a2i)
    Y = kde(np.vstack([A1.ravel(), A2.ravel()])).reshape(A1.shape)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    
    # Plot both the density and scatter points
    ax.contourf(A1, A2, Y, levels=levels, cmap=cmap, alpha=alpha)
    # ax.scatter(a1, a2, c='k', s=1, alpha=0.1)  # Add points overlay
    
    axis_names = ['X', 'Y', 'Z']
    ax.set_xlabel(f'{axis_names[axis1]} (m)', fontsize=36)
    ax.set_ylabel(f'{axis_names[axis2]} (m)', fontsize=36)
    ax.tick_params(axis='both', which='major', labelsize=30)
    
    # Set axis limits explicitly
    ax.set_xlim(a1.min() - a1_margin, a1.max() + a1_margin)
    ax.set_ylim(a2.min() - a2_margin, a2.max() + a2_margin)
    
    return ax

# Load ALL 100 parquet files and plot combined heatmap
if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import glob
    
    # Load ALL parquet files from chunk-000 directory
    parquet_dir = 'datasets/iss_docking_images/data/chunk-000'
    parquet_files = sorted(glob.glob(os.path.join(parquet_dir, "episode_*.parquet")))
    print(f"Loading {len(parquet_files)} parquet files...")
    
    try:
        all_positions = []
        
        for i, parquet_file in enumerate(parquet_files):
            df = pd.read_parquet(parquet_file)
            # Extract only x, y, z from observation.state
            positions = np.stack(df['observation.state'].to_numpy())[:, :3]  # shape (N, 3)
            all_positions.append(positions)
            
            if i % 20 == 0:  # Progress update every 20 files
                print(f"Processed {i+1}/{len(parquet_files)} files...")
        
        # Combine all positions from all episodes
        combined_positions = np.vstack(all_positions)
        
        # Print overall data statistics
        print(f"\nCombined Data Statistics:")
        print(f"Total number of trajectory points: {len(combined_positions)}")
        print(f"Number of episodes: {len(parquet_files)}")
        print("\nRanges for each axis:")
        for i, axis in enumerate(['X', 'Y', 'Z']):
            print(f"{axis}: [{combined_positions[:, i].min():.6f}, {combined_positions[:, i].max():.6f}]")
        
        # Create separate plots for each axis
        fig, ax = plt.subplots(figsize=(10, 4))
        
        # Plot X-Y, X-Z, and Y-Z views using ALL trajectory data
        plot_density_from_positions(combined_positions, axis1=2, axis2=1, ax=ax)
        # plot_density_from_positions(combined_positions, axis1=0, axis2=2, ax=ax)
        # plot_density_from_positions(combined_positions, axis1=1, axis2=2, ax=ax)
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Could not load or plot parquet files: {e}")

fig, ax = plt.subplots(figsize=(10, 4))
plot_density(episodes, ax, axis1=2, axis2=1)  # Default: Z-Y

plt.tight_layout()
plt.show()

# Plot histograms of all waypoints
# plot_histogram(episodes)
