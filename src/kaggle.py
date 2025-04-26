import pandas as pd
import os

# Load summary_kaggle.csv
summary = pd.read_csv("results/summary_kaggle.csv")

# Prepare columns
summary["feature_num"] = summary["feature_num"].astype(str)
summary["tuned"] = summary["tuned"].astype(int)
summary["model"] = summary["model"].astype(str)

# Create output folder
os.makedirs("stats/anova", exist_ok=True)

# Create feature_engineered column
summary["feature_engineered"] = (summary["feature_num"] != "baseline").astype(int)

# Rename kaggle_score to accuracy for JASP
summary = summary.rename(columns={"kaggle_score": "accuracy"})

# Select only needed columns
jasp_ready = summary[["accuracy", "model", "feature_engineered", "tuned"]]

# Save JASP-ready dataset
jasp_ready.to_csv("stats/anova/jasp_ready_data_kaggle.csv", index=False)

print("✅ Saved Kaggle JASP-ready data to stats/anova/jasp_ready_data_kaggle.csv")
