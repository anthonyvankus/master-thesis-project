import os
import pandas as pd
import numpy as np

# Get absolute path to the CSV file relative to this script's location
script_dir = os.path.dirname(__file__)
csv_path = os.path.join(script_dir, "training.csv")

# Load the CSV file
df = pd.read_csv(csv_path, index_col = 0, sep=";", quotechar='"')

# Ensure proper types
df['PolNum'] = df['PolNum'].astype(str)

# Identify policy numbers that appear exactly twice
dup_counts = df['PolNum'].value_counts()
dups_twice = dup_counts[dup_counts == 2].index

# Filter only those rows (with exactly 2 appearances)
df_dups = df[df['PolNum'].isin(dups_twice)].copy()

# Rank occurrences within each PolNum group
df_dups['occurrence'] = df_dups.groupby('PolNum').cumcount()

# Split into first and second occurrences
df_first = df_dups[df_dups['occurrence'] == 0].set_index('PolNum')
df_second = df_dups[df_dups['occurrence'] == 1].set_index('PolNum')

# Add Exppdays from first to second occurrence
df_second['Exppdays'] += df_first['Exppdays']

# Drop the helper column
df_second = df_second.drop(columns='occurrence')

# Reset index if needed
df_second = df_second.reset_index()

#print("Modified DataFrame with only second occurrences and updated Exppdays:")
#print(df_second)

# Remove both original duplicate rows from full df
df_cleaned = df[~df['PolNum'].isin(dups_twice)]

# Append the adjusted second occurrence rows
df_final = pd.concat([df_cleaned, df_second], ignore_index=True)

df_final['TotalNum'] = df_final['Numtppd'] + df_final['Numtpbi']

df_final['Exposure'] = df_final['Exppdays']/365

#print(df_final.head())

#filtered = df_final[(df_final['Numtppd'] != 0) & (df_final['Numtpbi'] != 0)]

#print("Rows where both Numtppd and Numtpbi are non-zero:")
#print(filtered)

df_final['AvIndtppd'] = np.where(df_final['Numtppd'] != 0,
                                 df_final['Indtppd'] / df_final['Numtppd'],
                                 0)

df_final['AvIndtpbi'] = np.where(df_final['Numtpbi'] != 0,
                                 df_final['Indtpbi'] / df_final['Numtpbi'],
                                 0)

# Get unique categories
#unique_occupation = df_final['Occupation'].unique()
#print("Unique occupations:", unique_occupation)

# Count how many unique categories
#num_occupation = df_final['Occupation'].nunique()
#print(f"Number of categories in 'Occupation': {num_occupation}")

# Get unique categories
#unique_type = df_final['Type'].unique()
#print("Unique categories:", unique_type)

# Count how many unique categories
#num_type = df_final['Type'].nunique()
#print(f"Number of categories in 'Type': {num_type}")

# Get unique categories
#unique_car_group = df_final['Group1'].unique()
#print("Unique categories:", unique_car_group)

# Count how many unique categories
#num_car_group = df_final['Group1'].nunique()
#print(f"Number of categories in 'Group1': {num_car_group}")

# Get unique categories
#unique_region_group = df_final['Group2'].unique()
#print("Unique categories:", unique_region_group)

# Count how many unique categories
#num_region_group = df_final['Group2'].nunique()
#print(f"Number of categories in 'Group2': {num_region_group}")

# Get unique categories
#unique_region_subgroup = df_final['SubGroup2'].unique()
#print("Unique categories:", unique_region_subgroup)

# Count how many unique categories
#num_region_subgroup = df_final['SubGroup2'].nunique()
#print(f"Number of categories in 'SubGroup2': {num_region_subgroup}")

df_encoded = pd.get_dummies(df_final, columns=['Gender', 'Type', 'Category', 'Occupation', 'Group1', 'Group2'])

# Convert all boolean columns in df_encoded to integers
df_encoded = df_encoded.astype({col: int for col in df_encoded.columns if df_encoded[col].dtype == bool})

pg15_training = pd.DataFrame(df_encoded)
