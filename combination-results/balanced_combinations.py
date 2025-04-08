# type: ignore
import pandas as pd

# File paths for each model
file_paths = {
    "Decision Tree": "1_dt_comb.csv",
    "XGBoost": "2_xgb_comb.csv",
    "Random Forest": "3_rf_comb.csv",
    "LightGBM": "4_lgbm_comb.csv",
    "CatBoost": "5_cb_comb.csv",
}

# Feature name to number map
FEATURE_MAP = {
    1: "title",
    2: "family_size",
    3: "is_alone",
    4: "age_group",
    5: "fare_per_person",
    6: "deck",
    7: "has_cabin",
    8: "is_mother",
    9: "sex_pclass",
    10: "is_child",
    11: "women_children_first",
}
NAME_TO_NUM = {v: k for k, v in FEATURE_MAP.items()}

# Model sort order
model_order = {
    "Decision Tree": 0,
    "XGBoost": 1,
    "Random Forest": 2,
    "LightGBM": 3,
    "CatBoost": 4,
}

# Process each model
balanced_samples = []

for model, path in file_paths.items():
    df = pd.read_csv(path).sort_values("accuracy_vs_kaggle").reset_index(drop=True)
    total = len(df)

    # Select top 3, middle 4, bottom 3
    top = df.tail(3)
    middle = df.iloc[(total - 4) // 2 : (total - 4) // 2 + 4]
    bottom = df.head(3)

    # Combine and annotate
    sample = pd.concat([top, middle, bottom], ignore_index=True)
    sample["model_name"] = model

    # Convert feature names to numbers
    sample["feature_nums"] = sample["features"].apply(
        lambda x: sorted(
            [NAME_TO_NUM[f.strip()] for f in x.split(",") if f.strip() in NAME_TO_NUM]
        )
    )

    balanced_samples.append(sample)

# Combine all samples
final_df = pd.concat(balanced_samples, ignore_index=True)

# Add sort helper columns
final_df["model_order"] = final_df["model_name"].map(model_order)
final_df["feature_len"] = final_df["feature_nums"].apply(len)
final_df["feature_str"] = final_df["feature_nums"].apply(
    lambda x: ",".join(map(str, x))
)

# Sort by model, then feature count, then feature values
final_df = final_df.sort_values(
    by=["model_order", "feature_len", "feature_str"]
).reset_index(drop=True)

# Drop helper columns
final_df = final_df.drop(columns=["model_order", "feature_len", "feature_str"])

# Show or save
print(final_df)
final_df.to_csv("balanced_combinations.csv", index=False)
