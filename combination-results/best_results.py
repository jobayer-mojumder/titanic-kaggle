# type: ignore
import pandas as pd

# Define file paths for each model
file_paths = {
    "Decision Tree": "1_dt_comb.csv",
    "XGBoost": "2_xgb_comb.csv",
    "Random Forest": "3_rf_comb.csv",
    "LightGBM": "4_lgbm_comb.csv",
    "CatBoost": "5_cb_comb.csv",
}

# Store best result rows
best_results = []

# Loop through each model and file
for model, path in file_paths.items():
    df = pd.read_csv(path)

    # Find the row with the highest accuracy
    best_row = df.loc[df["accuracy_vs_kaggle"].idxmax()].copy()
    best_row["model_name"] = model  # Add model label
    best_results.append(best_row)

# Create a DataFrame of best results
best_df = pd.DataFrame(best_results)

# Display result
print(best_df)

# Optional: Save to CSV
best_df.to_csv("best_results.csv", index=False)
