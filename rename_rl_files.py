import os
import pandas as pd
import glob

def rename_rl_files():
    """
    Rename RL filtered data files and update their run IDs sequentially.
    Files will be renamed from rl_analysis_run_0_*.csv to rl_analysis_run_1_*.csv, etc.
    Run IDs in each file will be updated to match the new file number.
    """
    
    # Path to the filtered data directory
    filtered_data_dir = "RL-Data/FilteredData"
    
    # Get all CSV files in the filtered data directory
    csv_files = glob.glob(os.path.join(filtered_data_dir, "*.csv"))
    
    # Sort files to ensure consistent ordering
    csv_files.sort()
    
    print(f"Found {len(csv_files)} CSV files to process:")
    for i, file in enumerate(csv_files):
        print(f"  {i+1}. {os.path.basename(file)}")
    
    # Process each file
    for run_id, file_path in enumerate(csv_files):
        try:
            # Read the CSV file
            df = pd.read_csv(file_path)
            
            # Update the Run ID column to the new run ID
            df['Run ID'] = run_id
            
            # Create new filename
            old_filename = os.path.basename(file_path)
            # Extract the timestamp part (everything after the first underscore)
            parts = old_filename.split('_', 3)  # Split on first 3 underscores
            if len(parts) >= 4:
                timestamp_part = '_'.join(parts[3:])  # Rejoin the timestamp part
                new_filename = f"rl_analysis_run_{run_id}_{timestamp_part}"
            else:
                # Fallback if filename structure is different
                new_filename = f"rl_analysis_run_{run_id}.csv"
            
            new_file_path = os.path.join(filtered_data_dir, new_filename)
            
            # Save the updated data to the new filename
            df.to_csv(new_file_path, index=False)
            
            # Remove the old file
            os.remove(file_path)
            
            print(f"✓ Processed: {old_filename} → {new_filename} (Run ID: {run_id})")
            
        except Exception as e:
            print(f"✗ Error processing {file_path}: {e}")
    
    print(f"\nCompleted! Processed {len(csv_files)} files.")
    print("All files have been renamed and their Run IDs updated.")

if __name__ == "__main__":
    rename_rl_files() 