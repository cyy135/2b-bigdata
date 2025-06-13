import pandas as pd
import numpy as np
import csv

def import_csv_with_pandas(file_path):
    """
    Import CSV data using pandas where each row has 4-dimensional values.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        numpy.ndarray: Array of shape (num_rows, 4) containing the 4D data
    """
    try:
        # Read CSV file - assuming no header and 4 columns
        df = pd.read_csv(file_path, header=None)
        
        # Check if we have exactly 4 columns
        if df.shape[1] != 4:
            print(f"Warning: Expected 4 columns, but found {df.shape[1]} columns")
            print("Using first 4 columns if available...")
            df = df.iloc[:, :4]  # Take first 4 columns
        
        # Convert to numpy array
        data = df.values
        
        print(f"Successfully imported {data.shape[0]} rows with {data.shape[1]} dimensions")
        print(f"Data shape: {data.shape}")
        print(f"First 5 rows:\n{data[:5]}")
        
        return data
        
    except Exception as e:
        print(f"Error importing with pandas: {e}")
        return None

def import_csv_with_csv_module(file_path):
    """
    Import CSV data using built-in csv module where each row has 4-dimensional values.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        list: List of lists, where each inner list contains 4 values
    """
    try:
        data = []
        with open(file_path, newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row_num, row in enumerate(reader):
                if len(row) < 4:
                    print(f"Warning: Row {row_num + 1} has fewer than 4 values: {row}")
                    continue
                
                # Convert each value to float (or int, as needed)
                try:
                    row_data = [float(value) for value in row[:4]]  # Take first 4 values
                    data.append(row_data)
                except ValueError as e:
                    print(f"Warning: Could not convert row {row_num + 1} to numbers: {row}")
                    continue
        
        print(f"Successfully imported {len(data)} rows with 4 dimensions")
        print(f"First 5 rows:\n{data[:5]}")
        
        return data
        
    except Exception as e:
        print(f"Error importing with csv module: {e}")
        return None

def main():
    """
    Main function to demonstrate importing CSV data with 4-dimensional values.
    """
    # List of CSV files to import
    csv_files = ['public_data.csv', 'private_data.csv']
    
    for file_path in csv_files:
        print(f"\n{'='*50}")
        print(f"Processing file: {file_path}")
        print(f"{'='*50}")
        
        # Method 1: Using pandas (recommended for large files)
        print("\n1. Using pandas:")
        data_pandas = import_csv_with_pandas(file_path)
        
        # Method 2: Using csv module
        print("\n2. Using csv module:")
        data_csv = import_csv_with_csv_module(file_path)
        
        # Convert csv module result to numpy array for consistency
        if data_csv:
            data_csv_array = np.array(data_csv)
            print(f"\nConverted to numpy array shape: {data_csv_array.shape}")
        
        print(f"\n{'='*50}")

if __name__ == "__main__":
    main() 