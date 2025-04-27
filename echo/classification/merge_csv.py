#!/usr/bin/env python
import os
import argparse
import pandas as pd


def merge_csv_files(csv_files, output_path, ignore_index=True, sort=True, concat_method="outer", rename_columns=None):
    """
    Merge multiple CSV files into a single CSV file.
    
    Args:
        csv_files (list): List of paths to CSV files to merge
        output_path (str): Path to save the merged CSV file
        ignore_index (bool): If True, reset the index in the merged file
        sort (bool): If True, sort the columns in the merged file
        concat_method (str): Method for pandas concat - "inner" or "outer" join
        rename_columns (dict): Dictionary mapping old column names to new ones (e.g., {'class_id_pred': 'class_id'})
    
    Returns:
        pd.DataFrame: The merged DataFrame
    """
    if not csv_files:
        raise ValueError("No CSV files provided for merging")
    
    print(f"Merging {len(csv_files)} CSV files...")
    
    # Read all dataframes
    dfs = []
    for file_path in csv_files:
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} does not exist. Skipping.")
            continue
            
        try:
            df = pd.read_csv(file_path)
            print(f"Read {file_path}, shape: {df.shape}")
            dfs.append(df)
        except Exception as e:
            print(f"Error reading file {file_path}: {str(e)}")
    
    if not dfs:
        raise ValueError("No valid CSV files were loaded")
    
    # Concatenate all dataframes
    merged_df = pd.concat(dfs, ignore_index=ignore_index, sort=sort, join=concat_method)
    print(f"Merged DataFrame shape: {merged_df.shape}")
    
    # Rename columns if specified
    if rename_columns:
        # Check if all columns to rename exist in the DataFrame
        missing_columns = [col for col in rename_columns.keys() if col not in merged_df.columns]
        if missing_columns:
            print(f"Warning: The following columns to rename don't exist in the merged DataFrame: {missing_columns}")
            # Only rename columns that exist
            rename_dict = {k: v for k, v in rename_columns.items() if k in merged_df.columns}
        else:
            rename_dict = rename_columns
            
        if rename_dict:
            merged_df.rename(columns=rename_dict, inplace=True)
            print(f"Renamed columns: {rename_dict}")
    
    # Save the merged dataframe
    merged_df.to_csv(output_path, index=False)
    print(f"Merged CSV saved to {output_path}")
    
    return merged_df


def main():
    parser = argparse.ArgumentParser(description="Merge multiple CSV files into one")
    parser.add_argument(
        "--csv-files", 
        type=str, 
        nargs="+", 
        required=True, 
        help="Paths to CSV files to merge"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        required=True, 
        help="Path to save the merged CSV file"
    )
    parser.add_argument(
        "--no-ignore-index", 
        action="store_false", 
        dest="ignore_index",
        help="Don't reset index in the merged file"
    )
    parser.add_argument(
        "--sort", 
        action="store_true", 
        help="Sort the columns in the merged file"
    )
    parser.add_argument(
        "--join", 
        type=str, 
        default="outer", 
        choices=["inner", "outer"],
        help="Join method for merging: inner or outer"
    )
    parser.add_argument(
        "--drop-duplicates", 
        action="store_true", 
        help="Remove duplicate rows from merged file"
    )
    parser.add_argument(
        "--rename-columns",
        type=str,
        nargs="+",
        help="Rename columns in format 'old_name:new_name', e.g., 'class_id_pred:class_id'"
    )
    
    args = parser.parse_args()
    
    # Process column renaming dictionary
    rename_columns = None
    if args.rename_columns:
        try:
            rename_columns = {}
            for rename_pair in args.rename_columns:
                old_name, new_name = rename_pair.split(":")
                rename_columns[old_name.strip()] = new_name.strip()
            print(f"Will rename columns: {rename_columns}")
        except ValueError:
            print("Error: Column renaming arguments must be in format 'old_name:new_name'")
            return
    
    merged_df = merge_csv_files(
        args.csv_files,
        args.output,
        ignore_index=args.ignore_index,
        sort=args.sort,
        concat_method=args.join,
        rename_columns=rename_columns
    )
    
    if args.drop_duplicates:
        rows_before = len(merged_df)
        merged_df.drop_duplicates(inplace=True)
        rows_after = len(merged_df)
        print(f"Removed {rows_before - rows_after} duplicate rows")
        merged_df.to_csv(args.output, index=False)
    
    print(f"Successfully merged {len(args.csv_files)} CSV files into {args.output}")


if __name__ == "__main__":
    main()
