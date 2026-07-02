import pandas as pd

# 1. SET FILE NAME: Replace with your actual 1.5 million rows dataset file name
original_file_name = 'shopee_reviews.csv' 

print("Reading data... Please wait a moment.")

try:
    # 2. Read only the first 20,000 rows using 'nrows'
    df = pd.read_csv(original_file_name, nrows=20000)

    # 3. Save the subset to a new file
    df.to_csv('raw_data.csv', index=False)

    print("-" * 40)
    print("SUCCESS!")
    print(f"New file 'data_20k.csv' generated with {len(df)} rows.")
    print("-" * 40)

except FileNotFoundError:
    print("ERROR: Original file not found. Ensure your .csv file is in the same folder as this script.")
except Exception as e:
    print(f"ERROR: Something went wrong: {e}")