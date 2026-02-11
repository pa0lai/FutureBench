import pandas as pd
import json
import glob
import os
import shutil

# --- Configuration ---
LLM_REPORT_FILE = "llm_quality_report_v2.csv" # The CSV report from the LLM check
TRIPLETS_DIR = "data/" # Directory containing your original ..._triplets.jsonl files
OUTPUT_DIR = "filtered_data/" # NEW directory to save the filtered .jsonl files
# --- Configuration End ---

def load_llm_report(filepath):
    """Loads the LLM quality report CSV."""
    if not os.path.exists(filepath):
        print(f"Error: LLM report file not found at '{filepath}'")
        return None
    try:
        df = pd.read_csv(filepath)
        print(f"Successfully loaded LLM report: {filepath}")
        # Convert relevant columns to nullable boolean type for safe comparison
        bool_cols = [
            'contains_answer_ctx_true', 'is_valid_insuf_ctx_insuf',
            'contains_answer_ctx_false', 'is_contradictory_ctx_false' 
        ]
        # Also check error columns existence implicitly later
        
        for col in bool_cols:
            if col in df.columns:
                try:
                    # Convert potential strings 'True'/'False' etc.robustly
                    df[col] = df[col].map({'True': True, 'False': False, True: True, False: False}).astype(pd.BooleanDtype())
                except Exception as e:
                     print(f"Warning: Could not reliably convert column '{col}' to boolean. Check data. Error: {e}")
            else:
                 print(f"Warning: Expected boolean column '{col}' not found in report.")
                 
        return df
    except Exception as e:
        print(f"Error reading LLM report CSV: {e}")
        return None

def filter_item_ids(report_df):
    """Filters item_ids based on LLM quality checks."""
    if report_df is None:
        return set()

    print("\nFiltering for high-quality item_ids based on LLM checks:")
    print("Criteria:")
    print("  - ctx_true contains answer")
    print("  - ctx_insuf is valid (relevant & no answer)")
    print("  - ctx_false does NOT contain answer")
    print("  - ctx_false IS contradictory")
    print("  - No API errors during LLM check for any context")

    # Define required columns for checking
    required_bool_cols = [
        'contains_answer_ctx_true', 'is_valid_insuf_ctx_insuf',
        'contains_answer_ctx_false', 'is_contradictory_ctx_false'
    ]
    required_error_cols = [
        'error_ctx_true', 'error_ctx_false', 'error_ctx_insuf'
    ]

    # Check if all required columns exist
    missing_cols = [col for col in required_bool_cols + required_error_cols if col not in report_df.columns]
    if missing_cols:
        print(f"Error: LLM report is missing required columns: {missing_cols}")
        return set()

    # --- Apply Filters ---
    # Start with all items checked by LLM
    filtered_df = report_df.copy()

    # Filter out items with API errors (check if error columns are NOT null/NA)
    error_condition = filtered_df[required_error_cols].notna().any(axis=1)
    filtered_df = filtered_df[~error_condition]
    print(f"  - Kept {len(filtered_df)} items after removing items with API errors.")

    # Apply boolean criteria using .eq() for safety with nullable booleans
    if 'contains_answer_ctx_true' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['contains_answer_ctx_true'].eq(True)]
        print(f"  - Kept {len(filtered_df)} items where ctx_true contains answer.")
        
    if 'is_valid_insuf_ctx_insuf' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['is_valid_insuf_ctx_insuf'].eq(True)]
        print(f"  - Kept {len(filtered_df)} items where ctx_insuf is valid.")
        
    if 'contains_answer_ctx_false' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['contains_answer_ctx_false'].eq(False)]
        print(f"  - Kept {len(filtered_df)} items where ctx_false does NOT contain answer.")
        
    if 'is_contradictory_ctx_false' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['is_contradictory_ctx_false'].eq(True)]
        print(f"  - Kept {len(filtered_df)} items where ctx_false IS contradictory.")

    # Get the set of item_ids to keep
    ids_to_keep = set(filtered_df['item_id'].unique())
    
    print(f"\nIdentified {len(ids_to_keep)} high-quality item_ids to keep.")
    
    total_checked = len(report_df['item_id'].unique())
    filtered_out = total_checked - len(ids_to_keep)
    if total_checked > 0:
         print(f"Filtered out {filtered_out} items ({filtered_out/total_checked:.1%}) due to quality issues or API errors.")
    
    return ids_to_keep

def process_triplet_files(input_dir, output_dir, ids_to_keep):
    """Reads original triplets, filters based on ids_to_keep, writes to output_dir."""
    
    if not ids_to_keep:
        print("Error: No item_ids to keep. Cannot process triplet files.")
        return

    # Create output directory if it doesn't exist
    if os.path.exists(output_dir):
        print(f"Warning: Output directory '{output_dir}' already exists. Files might be overwritten.")
    else:
        try:
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        except OSError as e:
            print(f"Error: Could not create output directory '{output_dir}'. {e}")
            return
            
    input_files = glob.glob(os.path.join(input_dir, "*.jsonl"))
    if not input_files:
        print(f"Error: No _triplets.jsonl files found in '{input_dir}'")
        return

    print(f"\nProcessing {len(input_files)} original triplet files from '{input_dir}'...")
    
    total_lines_read = 0
    total_lines_kept = 0

    for input_filepath in input_files:
        base_filename = os.path.basename(input_filepath)
        # Add '_filtered' before the .jsonl extension
        output_filename = base_filename.replace(".jsonl", "_filtered.jsonl")
        output_filepath = os.path.join(output_dir, output_filename)
        
        lines_kept_in_file = 0
        lines_read_in_file = 0
        
        print(f"  Processing: {base_filename} -> {output_filename}")
        
        try:
            with open(input_filepath, 'r', encoding='utf-8') as infile, \
                 open(output_filepath, 'w', encoding='utf-8') as outfile:
                 
                for line_num, line in enumerate(infile):
                    lines_read_in_file += 1
                    try:
                        item = json.loads(line)
                        # Use the same logic as quality check script to get item_id if missing
                        item_id = item.get('item_id', f"{base_filename}_{line_num}") 
                        
                        # Check if this item_id should be kept
                        if item_id in ids_to_keep:
                            # Write the original line to the output file
                            outfile.write(line) 
                            lines_kept_in_file += 1
                            
                    except json.JSONDecodeError:
                        print(f"    Warning: Skipping line {line_num+1} in {base_filename} (JSON decode error)")
                        
            print(f"    - Read {lines_read_in_file} lines, Kept {lines_kept_in_file} lines.")
            total_lines_read += lines_read_in_file
            total_lines_kept += lines_kept_in_file
            
        except Exception as e:
            print(f"  Error processing file {base_filename}: {e}")

    print("\n--- Filtering Summary ---")
    print(f"Total lines read from original files: {total_lines_read}")
    print(f"Total lines kept (high-quality): {total_lines_kept}")
    if total_lines_read > 0:
        kept_percentage = (total_lines_kept / total_lines_read) * 100
        print(f"Kept {kept_percentage:.1f}% of the original data.")
    print(f"Filtered data saved in directory: '{output_dir}'")

# --- Main Execution ---
if __name__ == "__main__":
    llm_report_df = load_llm_report(LLM_REPORT_FILE)
    high_quality_ids = filter_item_ids(llm_report_df)
    
    if high_quality_ids: # Only proceed if we have IDs to keep
        process_triplet_files(TRIPLETS_DIR, OUTPUT_DIR, high_quality_ids)
    else:
        print("\nNo high-quality items identified. No filtered files were created.")