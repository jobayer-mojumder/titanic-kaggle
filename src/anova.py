import pandas as pd
import ast
import os

# Load summary_local.csv
summary = pd.read_csv("results/summary_local.csv")

# Prepare columns
summary["feature_num"] = summary["feature_num"].astype(str)
summary["tuned"] = summary["tuned"].astype(int)
summary["model"] = summary["model"].astype(str)

# Create output folder
os.makedirs("stats/anova", exist_ok=True)

# Expand rows: expand folds for baseline (tuned=0 and tuned=1), use mean for others
expanded_rows = []

for idx, row in summary.iterrows():
    model = row["model"]
    feature_num = row["feature_num"]
    tuned = row["tuned"]
    acc_mean = row["accuracy"]
    cv_scores_str = row.get("cv_scores", None)

    if feature_num == "baseline":
        # For baseline and baseline tuned, expand all CV folds
        if pd.notna(cv_scores_str) and isinstance(cv_scores_str, str):
            try:
                fold_scores = ast.literal_eval(cv_scores_str)
                for fold_idx, fold_score in enumerate(fold_scores, 1):
                    expanded_rows.append(
                        {
                            "model": model,
                            "feature_num": feature_num,
                            "tuned": tuned,
                            "fold": fold_idx,
                            "accuracy": fold_score,
                            "feature_engineered": 0,  # baseline = no FE
                        }
                    )
            except Exception as e:
                print(f"⚠️ Error parsing cv_scores at row {idx}: {e}")
                continue
    else:
        # For feature engineering and feature engineering + tuning, use only the mean
        expanded_rows.append(
            {
                "model": model,
                "feature_num": feature_num,
                "tuned": tuned,
                "fold": 0,
                "accuracy": acc_mean,
                "feature_engineered": int(feature_num != "baseline"),
            }
        )

# Create expanded DataFrame
expanded_df = pd.DataFrame(expanded_rows)

# Save JASP-ready dataset
expanded_df = expanded_df[["accuracy", "model", "feature_engineered", "tuned"]]
expanded_df.to_csv("stats/anova/jasp_ready_data.csv", index=False)
print("✅ Saved JASP-ready data to stats/anova/jasp_ready_data.csv")
