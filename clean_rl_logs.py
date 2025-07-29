#!/usr/bin/env python3
"""
RL Log Cleaner - Post-processing script for RL-unfiltered-data logs

Takes raw RL inference logs and formats them nicely for Excel analysis.
Zero changes to core gameplay systems - pure post-processing.

Creates organized folder structure:
  RL-Data/
    RL-unfiltered-data_*.csv     - Raw timestamped log files  
    FilteredData/                - Clean Excel-friendly files

Usage:
    python clean_rl_logs.py                       # Clean log.csv -> timestamped files in RL-Data/FilteredData/
    python clean_rl_logs.py input.csv             # Clean specific file -> timestamped files
    python clean_rl_logs.py --clean-all           # Clean ALL CSV files in RL-Data/ directory
    python clean_rl_logs.py --no-timestamp        # Save to rl_analysis.csv instead of timestamps
"""

import pandas as pd
import argparse
import re
import sys
import os
from datetime import datetime

def parse_humanoid_class(humanoid_str):
    """Parse '(zombie, healthy)' -> ['zombie', 'healthy'] but handle multiple per side"""
    if pd.isna(humanoid_str) or humanoid_str == '(, )':
        return ['', '']
    
    # Remove parentheses and split by comma
    clean_str = humanoid_str.strip('()')
    parts = [part.strip() for part in clean_str.split(',') if part.strip()]
    
    if not parts:
        return ['', '']
    elif len(parts) == 1:
        return [parts[0], '']
    elif len(parts) == 2:
        return [parts[0], parts[1]]
    else:
        # Multiple humanoids - we need to make educated guess about left/right split
        # Assume roughly even split, with left side getting extra if odd number
        mid = (len(parts) + 1) // 2
        left_side = '|'.join(parts[:mid])
        right_side = '|'.join(parts[mid:])
        return [left_side, right_side]

def parse_capacity(capacity_str):
    """Parse '(2,10)' -> [2, 10]"""
    if pd.isna(capacity_str):
        return [0, 10]
    
    # Remove parentheses and split
    clean_str = capacity_str.strip('()')
    parts = clean_str.split(',')
    
    try:
        current = int(parts[0]) if parts[0] else 0
        max_cap = int(parts[1]) if len(parts) > 1 and parts[1] else 10
        return [current, max_cap]
    except:
        return [0, 10]

def parse_role(role_str):
    """Parse '(Civilian, Doctor)' -> ['Civilian', 'Doctor'] but handle multiple per side"""
    if pd.isna(role_str) or role_str == '(, )':
        return ['', '']
    
    # Remove parentheses and split by comma
    clean_str = role_str.strip('()')
    parts = [part.strip() for part in clean_str.split(',') if part.strip()]
    
    if not parts:
        return ['', '']
    elif len(parts) == 1:
        return [parts[0], '']
    elif len(parts) == 2:
        return [parts[0], parts[1]]
    else:
        # Multiple roles - split evenly like humanoids
        mid = (len(parts) + 1) // 2
        left_side = '|'.join(parts[:mid])
        right_side = '|'.join(parts[mid:])
        return [left_side, right_side]

def parse_inspected(inspected_str):
    """Parse 'NY' -> ['N', 'Y'] (left_inspected, right_inspected)"""
    if pd.isna(inspected_str) or len(inspected_str) != 2:
        return ['N', 'N']
    
    return [inspected_str[0], inspected_str[1]]

def create_situation_description(row):
    """Create a human-readable description of what happened in this step"""
    action = row['action']
    side = row['action_side']
    left = row['left_humanoids']
    right = row['right_humanoids']
    
    if action == 'save':
        side_text = "LEFT" if side == 'left' else "RIGHT"
        saved_people = left if side == 'left' else right
        return f"SAVED {saved_people} on {side_text} side"
    elif action == 'squish':
        side_text = "LEFT" if side == 'left' else "RIGHT"
        squished_people = left if side == 'left' else right
        return f"SQUISHED {squished_people} on {side_text} side"
    elif action == 'skip':
        return f"SKIPPED - Left: {left}, Right: {right}"
    else:
        return f"{action.upper()} - Left: {left}, Right: {right}"

def clean_log_data(df):
    """Convert raw log data to clean, Excel-friendly format"""
    # Extract final score if present
    final_score_rows = df[df['humanoid_class'] == 'FINAL_SCORE']
    final_score = None
    if not final_score_rows.empty:
        score_text = final_score_rows.iloc[0]['role']
        if score_text and score_text.startswith('SCORE:'):
            try:
                final_score = float(score_text.split('SCORE:')[1])
            except:
                final_score = None
    
    # Filter out empty/meaningless entries AND final score entries for main analysis
    meaningful_mask = ~(
        (df['humanoid_class'] == '(, )') | 
        (df['humanoid_class'].isna()) |
        (df['humanoid_class'] == 'FINAL_SCORE') |  # Exclude final score from main data
        (df['action'] == 'skip') & (df['humanoid_class'] == '(, )')
    )
    
    df_filtered = df[meaningful_mask].copy()
    
    if df_filtered.empty:
        print("❌ No meaningful actions found after filtering")
        return pd.DataFrame()  # Return empty DataFrame
    
    # Parse complex fields into clean columns
    print("🔧 Parsing complex fields...")
    
    # Parse humanoid classes
    humanoid_parsed = df_filtered['humanoid_class'].apply(parse_humanoid_class)
    df_filtered['left_humanoids'] = [x[0] for x in humanoid_parsed]
    df_filtered['right_humanoids'] = [x[1] for x in humanoid_parsed]
    
    # Parse roles  
    role_parsed = df_filtered['role'].apply(parse_role)
    df_filtered['left_roles'] = [x[0] for x in role_parsed]
    df_filtered['right_roles'] = [x[1] for x in role_parsed]
    
    # Parse capacity
    capacity_parsed = df_filtered['capacity'].apply(parse_capacity)
    df_filtered['people_in_ambulance'] = [x[0] for x in capacity_parsed]
    df_filtered['ambulance_capacity'] = [x[1] for x in capacity_parsed]
    
    # Parse inspection status
    inspected_parsed = df_filtered['inspected'].apply(parse_inspected)
    df_filtered['left_inspected'] = [x[0] for x in inspected_parsed]
    df_filtered['right_inspected'] = [x[1] for x in inspected_parsed]
    
    # Clean up movement numbers and add step counter
    df_filtered['step_number'] = range(1, len(df_filtered) + 1)
    
    # Create situation descriptions
    df_filtered['situation_description'] = df_filtered.apply(create_situation_description, axis=1)
    
    # Create final clean DataFrame with organized, human-readable columns
    clean_df = pd.DataFrame({
        # Step tracking
        'Step': df_filtered['step_number'],
        'Run ID': df_filtered['local_run_id'],
        
        # What happened (main info)
        'Situation': df_filtered['situation_description'],
        
        # Detailed breakdown
        'Left Side Had': df_filtered['left_humanoids'],
        'Left Side Roles': df_filtered['left_roles'], 
        'Right Side Had': df_filtered['right_humanoids'],
        'Right Side Roles': df_filtered['right_roles'],
        
        # RL decision
        'RL Action': df_filtered['action'],
        'Action Side': df_filtered['action_side'],
        
        # Game state
        'People in Ambulance': df_filtered['people_in_ambulance'],
        'Time Left (min)': df_filtered['remaining_time'],
        
        # Inspection status
        'Left Inspected': df_filtered['left_inspected'],
        'Right Inspected': df_filtered['right_inspected'],
    })
    
    # Add final score as metadata if available
    if final_score is not None:
        # Add final score as a summary row at the end
        final_row = pd.DataFrame({
            'Step': ['FINAL'],
            'Run ID': [clean_df['Run ID'].iloc[0] if not clean_df.empty else 0],
            'Situation': [f'🏆 FINAL SCORE: {final_score} points'],
            'Left Side Had': [''],
            'Left Side Roles': [''], 
            'Right Side Had': [''],
            'Right Side Roles': [''],
            'RL Action': ['game_end'],
            'Action Side': [''],
            'People in Ambulance': [clean_df['People in Ambulance'].iloc[-1] if not clean_df.empty else 0],
            'Time Left (min)': [clean_df['Time Left (min)'].iloc[-1] if not clean_df.empty else 0],
            'Left Inspected': [''],
            'Right Inspected': [''],
        })
        clean_df = pd.concat([clean_df, final_row], ignore_index=True)
    
    # Sort by step for chronological analysis (but keep FINAL at end)
    if not clean_df.empty and clean_df['Step'].iloc[-1] == 'FINAL':
        main_df = clean_df[clean_df['Step'] != 'FINAL'].sort_values('Step').reset_index(drop=True)
        final_df = clean_df[clean_df['Step'] == 'FINAL']
        clean_df = pd.concat([main_df, final_df], ignore_index=True)
    else:
        clean_df = clean_df.sort_values('Step').reset_index(drop=True)
    
    return clean_df

def clean_rl_logs(input_file, create_timestamped_files=True):
    """Main function to clean RL logs with organized folder structure"""
    try:
        # Create logs folder structure
        os.makedirs('RL-Data', exist_ok=True)
        os.makedirs('RL-Data/FilteredData', exist_ok=True)
        
        # Read raw log
        df = pd.read_csv(input_file)
        print(f"📊 Found {len(df)} entries in raw log")
        
        # Clean the data
        clean_df = clean_log_data(df)
        print(f"🧹 Cleaned to {len(clean_df)} meaningful actions")
        
        # Handle timestamped vs non-timestamped output
        if create_timestamped_files:
            # Extract timestamp from original filename if it exists
            filename_match = re.search(r'RL-unfiltered-data_(\d{8}_\d{6})', input_file)
            if filename_match:
                timestamp = filename_match.group(1)
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Archive original raw log in RL-Data (if it's not already there)
            if not input_file.startswith('RL-Data/'):
                raw_filename = f"RL-Data/RL-unfiltered-data_{timestamp}.csv"
                df.to_csv(raw_filename, index=False)
                print(f"📁 Raw log archived to {raw_filename}")
            
            # Save cleaned data with run info in filename
            run_count = len(clean_df['Run ID'].unique())
            if run_count > 0:
                run_ids = sorted(clean_df['Run ID'].unique())
                run_suffix = f"run_{run_ids[0]}" if run_count == 1 else f"runs_{run_ids[0]}-{run_ids[-1]}"
                clean_filename = f"RL-Data/FilteredData/rl_analysis_{run_suffix}_{timestamp}.csv"
            else:
                clean_filename = f"RL-Data/FilteredData/rl_analysis_{timestamp}.csv"
        else:
            clean_filename = "rl_analysis.csv"
            
        # Save cleaned data
        clean_df.to_csv(clean_filename, index=False)
        
        print(f"✅ Clean log saved to {clean_filename}")
        print(f"📈 Summary: {len(clean_df)} meaningful actions across {clean_df['Run ID'].nunique()} runs")
        
        # Quick analysis
        print("\n📊 Quick Analysis:")
        # Check if we have final score info
        final_score_row = clean_df[clean_df['Step'] == 'FINAL']
        if not final_score_row.empty:
            final_score_text = final_score_row.iloc[0]['Situation']
            print(f"   🏆 {final_score_text}")
        
        # Only analyze non-FINAL rows for action statistics
        action_df = clean_df[clean_df['Step'] != 'FINAL']
        if not action_df.empty:
            print(f"   • Most common action: {action_df['RL Action'].mode().iloc[0]}")
            print(f"   • Actions by side: {action_df['Action Side'].value_counts().to_dict()}")
            print(f"   • Avg people in ambulance: {action_df['People in Ambulance'].mean():.1f}")
            print(f"   • Steps per run: {len(action_df) / action_df['Run ID'].nunique():.1f}")
        else:
            print("   • No action data to analyze")
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing logs: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Clean RL inference logs for Excel analysis')
    parser.add_argument('input_file', nargs='?', default='log.csv', 
                       help='Input log file or directory (default: log.csv)')
    parser.add_argument('--no-timestamp', action='store_true',
                       help='Save to rl_analysis.csv instead of timestamped file')
    parser.add_argument('--clean-all', action='store_true',
                       help='Clean all CSV files in RL-Data/ directory')
    
    args = parser.parse_args()
    
    print(f"🧹 RL Log Cleaner - Making logs Excel-friendly!")
    
    # Determine input files to process
    if args.clean_all:
        # Process all CSV files in RL-Data/
        raw_dir = 'RL-Data'
        if not os.path.exists(raw_dir):
            print(f"❌ Directory {raw_dir} doesn't exist")
            sys.exit(1)
        
        csv_files = [f for f in os.listdir(raw_dir) if f.endswith('.csv')]
        if not csv_files:
            print(f"❌ No CSV files found in {raw_dir}")
            sys.exit(1)
        
        input_files = [os.path.join(raw_dir, f) for f in csv_files]
        print(f"📁 Processing {len(input_files)} files from {raw_dir}/")
    else:
        # Single file processing
        input_files = [args.input_file]
        print(f"📁 Input: {args.input_file}")
    
    if args.no_timestamp:
        print(f"📁 Output: rl_analysis.csv (single file)")
    else:
        print(f"📁 Output: Timestamped files in RL-Data/FilteredData/ folder")
        
    print("="*50)
    
    # Process each file
    all_success = True
    for input_file in input_files:
        print(f"\n🔄 Processing {input_file}...")
        success = clean_rl_logs(input_file, create_timestamped_files=not args.no_timestamp)
        if not success:
            all_success = False
    
    if all_success:
        print("="*50)
        print("🎉 Log cleaning completed successfully!")
        if not args.no_timestamp:
            print("💡 Check RL-Data/FilteredData/ folder for your Excel-ready files")
            print("💡 Raw logs are stored in RL-Data/ folder")
        else:
            print("💡 Open rl_analysis.csv in Excel for analysis")
    else:
        print("💥 Some log cleaning failed")
        sys.exit(1)

if __name__ == "__main__":
    main()