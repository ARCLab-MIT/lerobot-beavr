#!/usr/bin/env python

"""
Script to inspect the structure of a parquet file
"""

import argparse
import pandas as pd
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Inspect a parquet file")
    parser.add_argument("file_path", type=str, help="Path to the parquet file")
    args = parser.parse_args()
    
    # Read the parquet file
    df = pd.read_parquet(args.file_path)
    
    # Print basic information
    print(f"File: {args.file_path}")
    print(f"Number of rows: {len(df)}")
    print(f"Number of columns: {len(df.columns)}")
    print("\nColumns:")
    for col in df.columns:
        print(f"  - {col}")
    
    # Print sample data for each column
    print("\nSample data (first row):")
    first_row = df.iloc[0]
    for col, value in first_row.items():
        if isinstance(value, np.ndarray):
            print(f"  - {col}: numpy array with shape {value.shape} and values {value}")
        else:
            print(f"  - {col}: {value}")
    
    # Print data types
    print("\nData types:")
    for col, dtype in df.dtypes.items():
        print(f"  - {col}: {dtype}")

if __name__ == "__main__":
    main() 