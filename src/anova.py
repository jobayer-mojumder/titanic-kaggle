import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import ast
import os

# Load summary_local.csv
summary = pd.read_csv("results/summary_local.csv")

# Expand cv_scores into per-fold rows
expanded_rows = []

for idx, row in summary.iterrows():
    model = row["model"]
    feature_num = row["feature_num"]
    tuned = row["tuned"]
    cv_scores_str = row.get("cv_scores", None)

    if pd.isna(cv_scores_str) or not isinstance(cv_scores_str, str):
        continue

    try:
        fold_scores = ast.literal_eval(cv_scores_str)
    except Exception as e:
        print(f"⚠️ Error parsing cv_scores at row {idx}: {e}")
        continue

    for fold_idx, fold_score in enumerate(fold_scores, 1):
        expanded_rows.append(
            {
                "model": model,
                "feature_num": feature_num,
                "tuned": tuned,
                "fold": fold_idx,
                "accuracy": fold_score,
            }
        )

# Create expanded fold DataFrame
expanded_df = pd.DataFrame(expanded_rows)

# Prepare columns
expanded_df["feature_num"] = expanded_df["feature_num"].astype(str)
expanded_df["feature_engineered"] = (expanded_df["feature_num"] != "baseline").astype(
    int
)
expanded_df["tuned"] = expanded_df["tuned"].astype(int)
expanded_df["model"] = expanded_df["model"].astype(str)

# Create output folder
os.makedirs("stats/anova", exist_ok=True)

# List of your models
your_models = ["dt", "rf", "xgb", "lgbm", "cb"]


# Helper to classify effect size
def effect_size_label(eta_sq):
    if eta_sq < 0.01:
        return "Negligible"
    elif eta_sq < 0.06:
        return "Small"
    elif eta_sq < 0.14:
        return "Medium"
    else:
        return "Large"


# Prepare summary results
overall_results = []

# Prepare JASP-ready dataset
jasp_data = []

# Loop over each model
for model_key in your_models:
    model_data = expanded_df[expanded_df["model"] == model_key]

    comparisons = {
        "Baseline vs Feature Engineering": ("baseline", 0),
        "Baseline vs Model Tuning": ("baseline", 1),
        "Baseline vs FE+MT": ("!baseline", 1),
    }

    for name, (compare_feature, compare_tuned) in comparisons.items():
        baseline = model_data[
            (model_data["feature_num"] == "baseline") & (model_data["tuned"] == 0)
        ]

        if compare_feature == "baseline":
            compare = model_data[
                (model_data["feature_num"] == "baseline")
                & (model_data["tuned"] == compare_tuned)
            ]
        else:
            compare = model_data[
                (model_data["feature_num"] != "baseline")
                & (model_data["tuned"] == compare_tuned)
            ]

        if len(baseline) == 0 or len(compare) == 0:
            print(f"⚠️ Skipping {model_key.upper()} - {name} (not enough data)")
            continue

        combined = pd.concat([baseline, compare])
        if combined["accuracy"].nunique() == 1:
            print(
                f"⚠️ Skipping {model_key.upper()} - {name} (no variance between groups)"
            )
            continue

        combined["group"] = ["baseline"] * len(baseline) + ["compare"] * len(compare)

        formula = "accuracy ~ C(group)"
        lm = smf.ols(formula, data=combined).fit()
        anova_table = sm.stats.anova_lm(lm, typ=2)

        # Calculate partial eta squared
        ss_effect = anova_table.loc["C(group)", "sum_sq"]
        ss_total = anova_table["sum_sq"].sum()
        eta_sq = ss_effect / ss_total

        overall_results.append(
            {
                "Model": model_key.upper(),
                "Comparison": name,
                "F-Statistic": round(anova_table.loc["C(group)", "F"], 4),
                "p-value": round(anova_table.loc["C(group)", "PR(>F)"], 4),
                "Partial Eta Squared": round(eta_sq, 4),
                "Effect Size": effect_size_label(eta_sq),
            }
        )

    # Add to JASP data
    jasp_data.append(model_data[["accuracy", "model", "feature_engineered", "tuned"]])

# Save comparison results
overall_df = pd.DataFrame(overall_results)
overall_df.to_csv("stats/anova/comparison_results.csv", index=False)
print("✅ Saved all model comparison results to stats/anova/comparison_results.csv")

# Save JASP-ready dataset
jasp_ready = pd.concat(jasp_data, ignore_index=True)
jasp_ready.to_csv("stats/anova/jasp_ready_data.csv", index=False)
print("✅ Saved JASP-ready data to stats/anova/jasp_ready_data.csv")
