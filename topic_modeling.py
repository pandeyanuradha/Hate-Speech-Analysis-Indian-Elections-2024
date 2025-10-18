import pandas as pd
import glob
from bertopic import BERTopic

# ==============================================
# Topic Modeling Script for Facebook Election Data
# ==============================================
# This script performs topic modeling using BERTopic
# across multiple years of election-related Facebook data.
# It trains one model per year and a combined model
# for all years up to 2024.
# ==============================================

# Define the range of years to analyze
years = [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]

# DataFrame to hold all years' data
df_all = pd.DataFrame()

for year in years:
    year = str(year)
    print(f"\n=== Processing year {year} ===")

    df_combined = pd.DataFrame()

    # Load all group posts for the year
    group_path = f"Groups/{year}/*.csv"
    for fname in glob.glob(group_path):
        df = pd.read_csv(fname, sep=';')
        df_combined = pd.concat([df_combined, df], ignore_index=True)

    # Load all page posts for the year
    page_path = f"Pages/{year}/*.csv"
    for fname in glob.glob(page_path):
        df = pd.read_csv(fname, sep=';')
        df_combined = pd.concat([df_combined, df], ignore_index=True)

    # Merge into full dataset
    df_all = pd.concat([df_all, df_combined], ignore_index=True)

    # Remove missing messages
    df_combined = df_combined.dropna(subset=["message"])
    if df_combined.empty:
        print(f"No valid messages found for {year}. Skipping.")
        continue

    # Fit BERTopic model for the current year
    print(f"Training BERTopic model for {year}...")
    model = BERTopic(verbose=True)
    msg_list = df_combined["message"].astype(str).tolist()
    model.fit_transform(msg_list)
    model.save(f"model_year_{year}", serialization="safetensors", save_ctfidf=True)
    print(f"Model for {year} saved successfully!")

# Train a combined model for all years (up to 2024)
print("\n=== Training combined model for all years (up to 2024) ===")
df_all = df_all.dropna(subset=["message"])
if not df_all.empty:
    model = BERTopic(verbose=True)
    msg_list_all = df_all["message"].astype(str).tolist()
    model.fit_transform(msg_list_all)
    model.save("model_all_years_combined", serialization="safetensors", save_ctfidf=True)
    print("Combined model saved successfully!")
else:
    print("No data available for combined model.")