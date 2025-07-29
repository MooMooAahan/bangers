import os
import pandas as pd
import glob
import re
from io import StringIO

def aggregate_all_data():
    """
    Aggregate all data from filtered files.
    Capture every line after the header and combine into a single CSV file.
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
    header = None
    
    # Process each file
    for run_id, file_path in enumerate(csv_files):
        try:
            # Read the file line by line with UTF-8 encoding
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Extract header from first file
            if header is None:
                header = lines[0].strip()
            
            # Get all lines after the header (skip line 1)
            data_lines = []
            for line in lines[1:]:  # Skip header
                line_stripped = line.strip()
                if line_stripped:  # Only include non-empty lines
                    data_lines.append(line_stripped)
            
            if data_lines:
                # Create a temporary CSV string with header and data
                temp_csv_content = header + '\n' + '\n'.join(data_lines)
                
                # Read as DataFrame
                df = pd.read_csv(StringIO(temp_csv_content))
                
                # Update the Run ID to be sequential
                df['Run ID'] = run_id
                
                all_dataframes.append(df)
                print(f"✓ Processed: {os.path.basename(file_path)} (Run ID: {run_id}, {len(df)} data rows)")
            else:
                print(f"⚠ Warning: {os.path.basename(file_path)} has no data rows after header")
            
        except Exception as e:
            print(f"✗ Error processing {file_path}: {e}")
    
    if all_dataframes:
        # Combine all dataframes
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        
        # Save to new CSV file
        output_file = "RL-Data/RL_Compiled_Results.csv"
        combined_df.to_csv(output_file, index=False)
        
        print(f"\n✓ Successfully created: {output_file}")
        print(f"  Total rows: {len(combined_df)}")
        print(f"  Total runs: {len(csv_files)}")
        print(f"  Average rows per run: {len(combined_df) / len(csv_files):.1f}")
        
        # Show first few rows as preview
        print(f"\nPreview of combined data:")
        print(combined_df.head())
        
        # Show summary of run IDs
        print(f"\nRun ID distribution:")
        run_counts = combined_df['Run ID'].value_counts().sort_index()
        for run_id, count in run_counts.items():
            print(f"  Run {run_id}: {count} rows")
        
    else:
        print("No data was processed successfully.")

if __name__ == "__main__":
    aggregate_all_data() 