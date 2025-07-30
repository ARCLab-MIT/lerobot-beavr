import os
import pandas as pd
import numpy as np
from typing import List, Tuple

def calculate_mass_loss(file_path: str) -> Tuple[float, str]:
    """
    Calculate mass loss for a single trajectory file.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        Tuple of (mass_loss, filename)
    """
    try:
        df = pd.read_csv(file_path)
        if 'mass' not in df.columns:
            print(f"Warning: 'mass' column not found in {file_path}")
            return 0.0, os.path.basename(file_path)
        
        initial_mass = df['mass'].iloc[0]
        final_mass = df['mass'].iloc[-1]
        mass_loss = initial_mass - final_mass
        return mass_loss, os.path.basename(file_path)
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return 0.0, os.path.basename(file_path)

def analyze_mass_loss(directory: str) -> None:
    """
    Analyze mass loss across all CSV files in the specified directory.
    
    Args:
        directory: Path to directory containing CSV files
    """
    # Get list of all CSV files
    try:
        csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    except Exception as e:
        print(f"Error accessing directory {directory}: {str(e)}")
        return

    if not csv_files:
        print(f"No CSV files found in {directory}")
        return

    # Calculate mass loss for each file
    mass_losses: List[Tuple[float, str]] = []
    for file_name in csv_files:
        file_path = os.path.join(directory, file_name)
        mass_loss, fname = calculate_mass_loss(file_path)
        mass_losses.append((mass_loss, fname))

    # Calculate statistics
    mass_loss_values = [loss for loss, _ in mass_losses]
    avg_mass_loss = np.mean(mass_loss_values)
    std_mass_loss = np.std(mass_loss_values)
    
    # Print results
    print("\nMass Loss Analysis Results:")
    print("-" * 50)
    print(f"Number of files analyzed: {len(csv_files)}")
    print(f"Average mass loss: {avg_mass_loss:.6f}")
    print(f"Standard deviation: {std_mass_loss:.6f}")
    print("\nIndividual file results:")
    print("-" * 50)
    
    # Sort by mass loss for better visualization
    mass_losses.sort(key=lambda x: x[0], reverse=True)
    for mass_loss, file_name in mass_losses:
        print(f"{file_name}: {mass_loss:.6f}")

if __name__ == "__main__":
    directory = "/home/demo/Alex/MetaRL Paper/docking_trajectories (2)/trajectories/states"
    analyze_mass_loss(directory)
