import os
import pandas as pd
import glob
import re
from io import StringIO

def extract_results():
    """
    Extract lines containing 'final' from all CSV files.
    Add them to a results file and remove from original files.
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
    
    # List to store all final results
    all_final_results = []
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
            
            # Find lines containing 'final' (case insensitive)
            final_lines = []
            remaining_lines = []
            
            for line in lines:
                line_stripped = line.strip()
                if line_stripped and 'final' in line_stripped.lower():
                    final_lines.append(line_stripped)
                else:
                    remaining_lines.append(line)
            
            if final_lines:
                # Create a temporary CSV string with header and final data
                temp_csv_content = header + '\n' + '\n'.join(final_lines)
                
                # Read as DataFrame
                df = pd.read_csv(StringIO(temp_csv_content))
                
                # Update the Run ID to be sequential
                df['Run ID'] = run_id
                
                all_final_results.append(df)
                print(f"✓ Processed: {os.path.basename(file_path)} (Run ID: {run_id}, {len(df)} final results)")
                
                # Write back the file without final lines
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.writelines(remaining_lines)
                print(f"  → Removed {len(final_lines)} final lines from original file")
            else:
                print(f"⚠ Warning: {os.path.basename(file_path)} has no lines containing 'final'")
            
        except Exception as e:
            print(f"✗ Error processing {file_path}: {e}")
    
    if all_final_results:
        # Combine all final results
        combined_results = pd.concat(all_final_results, ignore_index=True)
        
        # Save to results file
        results_file = "RL-Data/FilteredData/Results.csv"
        combined_results.to_csv(results_file, index=False)
        
        print(f"\n✓ Successfully created: {results_file}")
        print(f"  Total final results: {len(combined_results)}")
        print(f"  Total runs: {len(csv_files)}")
        
        # Show first few rows as preview
        print(f"\nPreview of final results:")
        print(combined_results.head())
        
        # Show summary of run IDs
        print(f"\nRun ID distribution:")
        run_counts = combined_results['Run ID'].value_counts().sort_index()
        for run_id, count in run_counts.items():
            print(f"  Run {run_id}: {count} final results")
        
    else:
        print("No final results were found.")

if __name__ == "__main__":
    extract_results() 