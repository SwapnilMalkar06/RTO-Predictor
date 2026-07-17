import pandas as pd
import os

# Define relative path structure to the Excel dataset location
file_path = os.path.join("data", "amazon_returns_dataset_cleaned.xlsx")

print("Checking data file accessibility...")
if not os.path.exists(file_path):
    print(f"❌ Error: Could not locate file at '{file_path}'. Check your file names inside the data folder.")
else:
    try:
        # Load the spreadsheet using openpyxl engine parsing
        df = pd.read_excel(file_path, engine="openpyxl")
        df.columns = df.columns.str.strip() # Remove any trailing or accidental spacing noise
        
        print("\n🎉 Excel Target File Read Successfully!")
        print(f"Total Transactions Tracked: {df.shape[0]}")
        print(f"Total Structural Features Identified: {df.shape[1]}")
        
        print("\n=== Exact Dataset Column Headers ===")
        for idx, col in enumerate(df.columns, 1):
            print(f" [{idx}] -> {col}")
            
        print("\n=== Data Row Visual Sneak Peek ===")
        print(df.iloc[0].to_dict())
        
    except Exception as e:
        print(f"❌ Unexpected Error executing file structural checks: {e}")