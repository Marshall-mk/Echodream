import pandas as pd
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description='Add class_id and class_name columns to CSV file based on CondValue')
parser.add_argument('--input', required=True, help='Input CSV file path')
parser.add_argument('--output', required=True, help='Output CSV file path')

# Parse arguments
args = parser.parse_args()

# Read the CSV file
df = pd.read_csv(args.input)

# Add class_id column based on CondValue
df['class_id'] = df['CondValue'].map({0: 0, 100: 1, 200: 2})

# Add class_name column based on class_id
df['class_name'] = df['class_id'].map({0: 'Normal', 1: 'ASD', 2: 'PAH'})

# Save the modified dataframe to a new CSV file
df.to_csv(args.output, index=False)

print(f"CSV file has been modified and saved as '{args.output}'")