import os
import pandas as pd
import glob
import re

def aggregate_rl_data_v2():
    """
    Aggregate RL data from all filtered files.
    Check if each line starts with a number and include those lines.
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
    header = None
    
    # Process each file
    for run_id, file_path in enumerate(csv_files):
        try:
            # Read the file line by line
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            # Extract header from first file
            if header is None:
                header = lines[0].strip()
            
            # Find lines that start with numbers (data rows)
            data_lines = []
            for line in lines[1:]:  # Skip header
                line = line.strip()
                if line and re.match(r'^\d+', line):  # Line starts with a number
                    data_lines.append(line)
            
            if data_lines:
                # Create a temporary CSV string with header and data
                temp_csv_content = header + '\n' + '\n'.join(data_lines)
                
                # Read as DataFrame
                df = pd.read_csv(pd.StringIO(temp_csv_content))
                
                # Update the Run ID to be sequential
                df['Run ID'] = run_id
                
                all_dataframes.append(df)
                print(f"✓ Processed: {os.path.basename(file_path)} (Run ID: {run_id}, {len(df)} data rows)")
            else:
                print(f"⚠ Warning: {os.path.basename(file_path)} has no data rows starting with numbers")
            
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
        
        # Show summary of run IDs
        print(f"\nRun ID distribution:")
        run_counts = combined_df['Run ID'].value_counts().sort_index()
        for run_id, count in run_counts.items():
            print(f"  Run {run_id}: {count} rows")
        
    else:
        print("No data was processed successfully.")

if __name__ == "__main__":
    aggregate_rl_data_v2() 