import pickle
import os
import numpy as np

PKL_PATH = os.path.join(os.path.dirname(__file__), 'unique_episodes.pkl')

def norm(arr):
    arr = np.array(arr)
    return np.linalg.norm(arr, axis=-1)

def main():
    with open(PKL_PATH, 'rb') as f:
        episodes = pickle.load(f)
    print(f"Loaded {len(episodes)} unique episodes from {PKL_PATH}\n")

    # Collect statistics
    start_positions = []
    end_positions = []
    durations = []
    mass_lost = []
    max_torque = []
    max_thrust = []
    start_masses = []
    end_masses = []
    delta_ts = []  # Store all delta time steps
    for idx, ep in enumerate(episodes):
        x = np.array(ep['x'])
        pos = x[:, :3]
        start_positions.append(pos[0])
        end_positions.append(pos[-1])
        # Trajectory duration
        t = np.array(ep['time'])
        durations.append(t[-1] - t[0])
        # Delta time steps
        if len(t) > 1:
            delta_t = np.diff(t)
            delta_ts.extend(delta_t)
        # Mass lost
        m = np.array(ep['mass'])
        start_masses.append(m[0])
        end_masses.append(m[-1])
        mass_lost.append(m[0] - m[-1])
        # Max torque (L)
        L = np.array(ep['L'])
        max_torque.append(np.max(norm(L)))
        # Max thrust (T)
        T = np.array(ep['T'])
        max_thrust.append(np.max(norm(T)))

    def summarize(arr, name):
        arr = np.array(arr)
        print(f"{name}: mean={arr.mean():.4f}, std={arr.std():.4f}, min={arr.min():.4f}, max={arr.max():.4f}")

    # Report delta time step statistics
    print("--- Delta Time Step Statistics (across all episodes) ---")
    delta_ts = np.array(delta_ts)
    if len(delta_ts) > 0:
        print(f"Delta t: mean={delta_ts.mean():.6f}, std={delta_ts.std():.6f}, min={delta_ts.min():.6f}, max={delta_ts.max():.6f}")
    else:
        print("No delta time steps found.")

    print("--- Start Position Statistics (per axis) ---")
    start_positions = np.array(start_positions)
    for i, axis in enumerate(['X', 'Y', 'Z']):
        summarize(start_positions[:, i], f"Start {axis}")
    print("\n--- End Position Statistics (per axis) ---")
    end_positions = np.array(end_positions)
    for i, axis in enumerate(['X', 'Y', 'Z']):
        summarize(end_positions[:, i], f"End {axis}")
    print("\n--- Trajectory Duration (s) ---")
    summarize(durations, "Duration")
    print("\n--- Mass Lost (start - end) ---")
    summarize(mass_lost, "Mass Lost")
    print("\n--- Start Mass ---")
    summarize(start_masses, "Start Mass")
    print("\n--- End Mass ---")
    summarize(end_masses, "End Mass")
    print("\n--- Max Torque Applied (norm of L) ---")
    summarize(max_torque, "Max Torque")
    print("\n--- Max Thrust Applied (norm of T) ---")
    summarize(max_thrust, "Max Thrust")

    # Optionally, print a table of all episodes
    print("\n--- Per-Episode Summary (first 5 episodes) ---")
    for i in range(min(20, len(episodes))):
        print(f"Episode {i}: Start {start_positions[i]}, End {end_positions[i]}, Duration {durations[i]:.2f}s, Mass Lost {mass_lost[i]:.4f}, Max Torque {max_torque[i]:.4f}, Max Thrust {max_thrust[i]:.4f}")

if __name__ == "__main__":
    main() 