import os
import pandas as pd
import glob

def aggregate_rl_data():
    """
    Aggregate RL data from all filtered files.
    Extract lines 2-22 from each file along with the header.
    Combine into a single CSV file with sequential run IDs.
    """
    
    # Path to the filtered data directory
    filtered_data_dir = "RL-Data/FilteredData"
    
    # Get all CSV files in the filtered data directory
    csv_files = glob.glob(os.path.join(filtered_data_dir, "*.csv"))
    
    # Sort files to ensure consistent ordering
    csv_files.sort()
    
    print(f"Found {len(csv_files)} CSV files to aggregate:")
    for i, file in enumerate(csv_files):
        print(f"  {i+1}. {os.path.basename(file)}")
    
    # List to store all dataframes
    all_dataframes = []
    
    # Process each file
    for run_id, file_path in enumerate(csv_files):
        try:
            # Read the CSV file
            df = pd.read_csv(file_path)
            
            # Extract lines 2-22 (rows 1-21 in 0-indexed pandas)
            # This gives us 21 data points per file
            if len(df) >= 21:
                df_subset = df.iloc[1:22].copy()  # Lines 2-22 (rows 1-21)
            else:
                # If file has fewer than 21 rows, take all available data
                df_subset = df.iloc[1:].copy()  # All rows except header
                print(f"  Warning: {os.path.basename(file_path)} has only {len(df)} rows")
            
            # Update the Run ID to be sequential
            df_subset['Run ID'] = run_id
            
            all_dataframes.append(df_subset)
            print(f"✓ Processed: {os.path.basename(file_path)} (Run ID: {run_id}, {len(df_subset)} rows)")
            
        except Exception as e:
            print(f"✗ Error processing {file_path}: {e}")
    
    if all_dataframes:
        # Combine all dataframes
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        
        # Save to new CSV file
        output_file = "RL-Data/RL_Full_Analysis.csv"
        combined_df.to_csv(output_file, index=False)
        
        print(f"\n✓ Successfully created: {output_file}")
        print(f"  Total rows: {len(combined_df)}")
        print(f"  Total runs: {len(csv_files)}")
        print(f"  Average rows per run: {len(combined_df) / len(csv_files):.1f}")
        
        # Show first few rows as preview
        print(f"\nPreview of combined data:")
        print(combined_df.head())
        
    else:
        print("No data was processed successfully.")

if __name__ == "__main__":
    aggregate_rl_data() 