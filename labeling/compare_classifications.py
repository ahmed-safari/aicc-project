#!/usr/bin/env python3
"""
Script to compare classifications from different sources.
Only keeps classifications where they match, otherwise leaves empty.
"""

import pandas as pd
import sys
from pathlib import Path

# Configuration
CONFIG = {
    'prefer_source': 'fanar',  # Options: 'gpt', 'fanar' - which source to prefer when one is empty
    'require_match': False,     # If True, only keep classifications when both sources agree; leave empty if they disagree
}

def compare_classifications(input_file, output_file, col1='mental_health_classification', col2='fanar_classification'):
    """
    Compare two classification columns and keep only matching classifications.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
        col1: Name of first classification column
        col2: Name of second classification column
    """
    
    print(f"Reading data from: {input_file}")
    df = pd.read_csv(input_file)
    
    print(f"\n{'='*60}")
    print(f"CONFIGURATION")
    print(f"{'='*60}")
    print(f"Preferred source: {CONFIG['prefer_source']}")
    print(f"Require match: {CONFIG['require_match']}")
    
    # Check if columns exist
    if col1 not in df.columns or col2 not in df.columns:
        print(f"Error: Columns '{col1}' and/or '{col2}' not found in the CSV file.")
        print(f"Available columns: {', '.join(df.columns)}")
        sys.exit(1)
    
    print(f"\nOriginal data shape: {df.shape}")
    print(f"\nColumn 1 ({col1}) value counts:")
    print(df[col1].value_counts())
    print(f"\nColumn 2 ({col2}) value counts:")
    print(df[col2].value_counts())
    
    # Create new column with matching classifications only
    def get_matching_classification(row):
        gpt_val = row[col1]
        fanar_val = row[col2]
        
        # Check if values are empty
        gpt_empty = pd.isna(gpt_val) or gpt_val == '' or str(gpt_val).strip() == ''
        fanar_empty = pd.isna(fanar_val) or fanar_val == '' or str(fanar_val).strip() == ''
        
        # If both are empty, return empty
        if gpt_empty and fanar_empty:
            return ''
        
        # If require_match is True, only keep when both agree or one is empty
        if CONFIG['require_match']:
            # Both have values - check if they match
            if not gpt_empty and not fanar_empty:
                if gpt_val == fanar_val:
                    return gpt_val
                else:
                    return ''  # They disagree, leave empty
            # One is empty - use the available one
            elif not fanar_empty:
                return fanar_val
            elif not gpt_empty:
                return gpt_val
            else:
                return ''
        
        # If require_match is False, use preferred source
        if CONFIG['prefer_source'] == 'fanar':
            # Prefer Fanar: if Fanar has value, use it; otherwise use GPT
            if not fanar_empty:
                return fanar_val
            elif not gpt_empty:
                return gpt_val
            else:
                return ''
        else:  # prefer_source == 'gpt'
            # Prefer GPT: if GPT has value, use it; otherwise use Fanar
            if not gpt_empty:
                return gpt_val
            elif not fanar_empty:
                return fanar_val
            else:
                return ''
    
    df['classification'] = df.apply(get_matching_classification, axis=1)
    
    # Calculate statistics
    total_rows = len(df)
    matching_rows = (df['classification'] != '').sum()
    non_matching_rows = (df['classification'] == '').sum()
    fanar_empty = df[col2].isna().sum() + (df[col2] == '').sum()
    gpt_empty = df[col1].isna().sum() + (df[col1] == '').sum()
    both_have_values = ((df[col1].notna()) & (df[col1] != '') & (df[col2].notna()) & (df[col2] != '')).sum()
    both_match = ((df[col1] == df[col2]) & (df[col1].notna()) & (df[col2].notna()) & (df[col2] != '') & (df[col1] != '')).sum()
    both_disagree = ((df[col1] != df[col2]) & (df[col1].notna()) & (df[col2].notna()) & (df[col2] != '') & (df[col1] != '')).sum()
    
    if CONFIG['prefer_source'] == 'fanar':
        used_fanar = ((df['classification'] == df[col2]) & (df[col2].notna()) & (df[col2] != '')).sum()
        used_gpt = ((df['classification'] == df[col1]) & ((df[col2].isna()) | (df[col2] == ''))).sum()
    else:
        used_gpt = ((df['classification'] == df[col1]) & (df[col1].notna()) & (df[col1] != '')).sum()
        used_fanar = ((df['classification'] == df[col2]) & ((df[col1].isna()) | (df[col1] == ''))).sum()
    
    print(f"\n{'='*60}")
    print(f"COMPARISON RESULTS")
    print(f"{'='*60}")
    print(f"Total rows: {total_rows:,}")
    print(f"GPT classifications empty: {gpt_empty:,} ({gpt_empty/total_rows*100:.2f}%)")
    print(f"Fanar classifications empty: {fanar_empty:,} ({fanar_empty/total_rows*100:.2f}%)")
    print(f"Both sources have values: {both_have_values:,} ({both_have_values/total_rows*100:.2f}%)")
    print(f"  - Both agree: {both_match:,} ({both_match/total_rows*100:.2f}%)")
    print(f"  - Both disagree: {both_disagree:,} ({both_disagree/total_rows*100:.2f}%)")
    print(f"Used Fanar: {used_fanar:,} ({used_fanar/total_rows*100:.2f}%)")
    print(f"Used GPT: {used_gpt:,} ({used_gpt/total_rows*100:.2f}%)")
    print(f"Final classifications kept: {matching_rows:,} ({matching_rows/total_rows*100:.2f}%)")
    print(f"Empty: {non_matching_rows:,} ({non_matching_rows/total_rows*100:.2f}%)")
    
    print(f"\nFinal classification distribution:")
    print(df['classification'].value_counts())
    
    # Prepare output dataframe
    # Keep all columns except the two original classification columns
    output_df = df.drop(columns=[col1, col2])
    
    # Save to CSV
    output_df.to_csv(output_file, index=False)
    print(f"\n✅ Saved results to: {output_file}")
    print(f"Output shape: {output_df.shape}")
    
    # Show some examples
    print(f"\n{'='*60}")
    print("SAMPLE MATCHES:")
    print(f"{'='*60}")
    matches = df[df['classification'] != ''].head(3)
    for idx, row in matches.iterrows():
        print(f"\nText: {row['text'][:80]}...")
        print(f"Classification: {row['classification']}")
    
    print(f"\n{'='*60}")
    print("SAMPLE NON-MATCHES:")
    print(f"{'='*60}")
    non_matches = df[df['classification'] == ''].head(3)
    for idx, row in non_matches.iterrows():
        print(f"\nText: {row['text'][:80]}...")
        print(f"{col1}: {row[col1]}")
        print(f"{col2}: {row[col2]}")
        print(f"Result: (empty)")

def main():
    # Default paths
    input_file = 'datasets/arabic_tweets_classified.csv'
    output_file = 'datasets/arabic_tweets_matched_classifications.csv'
    
    # Allow command line arguments
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    # Check if input file exists
    if not Path(input_file).exists():
        print(f"Error: Input file '{input_file}' not found.")
        sys.exit(1)
    
    # Run comparison
    compare_classifications(input_file, output_file)
    
    print(f"\n{'='*60}")
    print("✅ Classification comparison completed successfully!")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
