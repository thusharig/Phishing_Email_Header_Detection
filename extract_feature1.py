#  extract_feature.py
import pandas as pd
import os
import re
from urllib.parse import urlparse

#  Load your input file (ensure it exists)
df = pd.read_csv('data/processed/email_headers_processed1.csv', low_memory=False)

# Helper: Extract domain from 'sender' field
def extract_domain(from_field):
    match = re.search(r'@([\w.-]+)', str(from_field))
    return match.group(1).lower() if match else 'unknown'

#  Apply domain extraction
df['from_domain'] = df['sender'].apply(extract_domain)

#  Helper: Count number of Received headers
def count_received(header_text):
    return str(header_text).count('Received:')

#  Helper: Detect SPF failure
def detect_spf_fail(header_text):
    return int('spf=fail' in str(header_text).lower())

#  Header-based features (if column exists)
if 'header' in df.columns:
    df['received_count'] = df['header'].apply(count_received)
    df['spf_fail'] = df['header'].apply(detect_spf_fail)
else:
    df['received_count'] = 0
    df['spf_fail'] = 0

#  Date parsing and feature extraction
df['date'] = pd.to_datetime(df['date'], errors='coerce', utc=True)  # Fix timezone warning
df = df.dropna(subset=['date'])  # Remove rows where date parsing failed

# Time-based features
df['hour'] = df['date'].dt.hour
df['day_of_week'] = df['date'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
df['is_after_hours'] = ((df['hour'] < 8) | (df['hour'] > 18)).astype(int)

#  Save to CSV file
output_path = 'data/processed/email_features1.csv'
df.to_csv(output_path, index=False)

#  Preview
print(f"✅ Features saved to: {output_path}")
print(df[['from_domain', 'received_count', 'spf_fail', 'hour', 'day_of_week', 'is_weekend', 'is_after_hours']].head())
