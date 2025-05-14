# preprocess_email.py

import pandas as pd
import os

# Define file paths
input_path = 'data/raw/CEAS_08.csv'
output_dir = 'data/processed'
output_file = os.path.join(output_dir, 'email_headers_processed1.csv')

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

#  Load raw data
print("Loading raw dataset...")
try:
    df = pd.read_csv(input_path)
    print(f"✅ Loaded {len(df)} rows and {len(df.columns)} columns.")
except FileNotFoundError:
    print(f"❌ File not found: {input_path}")
    exit()

# Drop irrelevant columns
print(" Dropping irrelevant columns...")
columns_to_drop = ['Message-ID', 'X-Spam-Status']
df_cleaned = df.drop(columns=columns_to_drop, errors='ignore')

#  Handle missing values
print("Handling missing values...")
df_cleaned.fillna({'From': 'Unknown', 'To': 'Unknown'}, inplace=True)

#  Save cleaned data
df_cleaned.to_csv(output_file, index=False)
print(f"Cleaned data saved to: {output_file}")
print(f"🔍 Final shape: {df_cleaned.shape}")
