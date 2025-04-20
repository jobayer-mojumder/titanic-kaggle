import pandas as pd
import os
import scipy.stats as stats
from tabulate import tabulate

# Ensure output folder exists
os.makedirs("stats", exist_ok=True)

# Store all results here
anova_results = []


def run_anova(mode):
    path = f"results/summary_{mode}.csv"
    if not os.path.exists(path):
        print(f"⚠️ Skipping {mode.upper()} — summary file not found.")
        return

    summary = pd.read_csv(path)
    score_col = "accuracy" if mode == "local" else "kaggle_score"

    baseline = summary[
        (summary["feature_num"] == "baseline") & (summary["tuned"] == 0)
    ][score_col].tolist()
    tuning_only = summary[
        (summary["feature_num"] == "baseline") & (summary["tuned"] == 1)
    ][score_col].tolist()
    fe_only = summary[(summary["feature_num"] != "baseline") & (summary["tuned"] == 0)][
        score_col
    ].tolist()
    fe_tuning = summary[
        (summary["feature_num"] != "baseline") & (summary["tuned"] == 1)
    ][score_col].tolist()

    f_stat, p_value = stats.f_oneway(baseline, fe_only, tuning_only, fe_tuning)

    anova_results.append(
        {
            "Model": "ALL",
            "Mode": mode.upper(),
            "F-Statistic": f_stat,
            "P-Value": p_value,
            "Significant": "✅ Yes" if p_value < 0.05 else "❌ No",
        }
    )


def run_anova_per_model(mode):
    path = f"results/summary_{mode}.csv"
    if not os.path.exists(path):
        print(f"⚠️ Skipping {mode.upper()} per-model — summary file not found.")
        return

    summary = pd.read_csv(path)
    score_col = "accuracy" if mode == "local" else "kaggle_score"
    models = summary["model"].unique()

    for model in models:
        baseline = summary[
            (summary["model"] == model)
            & (summary["feature_num"] == "baseline")
            & (summary["tuned"] == 0)
        ][score_col].tolist()
        tuning_only = summary[
            (summary["model"] == model)
            & (summary["feature_num"] == "baseline")
            & (summary["tuned"] == 1)
        ][score_col].tolist()
        fe_only = summary[
            (summary["model"] == model)
            & (summary["feature_num"] != "baseline")
            & (summary["tuned"] == 0)
        ][score_col].tolist()
        fe_tuning = summary[
            (summary["model"] == model)
            & (summary["feature_num"] != "baseline")
            & (summary["tuned"] == 1)
        ][score_col].tolist()

        if any(len(g) == 0 for g in [baseline, fe_only, tuning_only, fe_tuning]):
            continue

        try:
            f_stat, p_value = stats.f_oneway(baseline, fe_only, tuning_only, fe_tuning)
            anova_results.append(
                {
                    "Model": model.upper(),
                    "Mode": mode.upper(),
                    "F-Statistic": f_stat,
                    "P-Value": p_value,
                    "Significant": "✅ Yes" if p_value < 0.05 else "❌ No",
                }
            )
        except Exception as e:
            anova_results.append(
                {
                    "Model": model.upper(),
                    "Mode": mode.upper(),
                    "F-Statistic": "ERROR",
                    "P-Value": "ERROR",
                    "Significant": str(e),
                }
            )


# Run all ANOVA
run_anova("local")
run_anova("kaggle")
run_anova_per_model("local")
run_anova_per_model("kaggle")

# Save and display if results exist
if anova_results:
    anova_df = pd.DataFrame(anova_results)
    anova_df["F-Statistic"] = anova_df["F-Statistic"].apply(
        lambda x: f"{float(x):8.4f}" if isinstance(x, (float, int)) else x
    )
    anova_df["P-Value"] = anova_df["P-Value"].apply(
        lambda x: f"{float(x):10.6f}" if isinstance(x, (float, int)) else x
    )
    anova_df.to_csv("stats/anova.csv", index=False)
    print(tabulate(anova_df, headers="keys", tablefmt="fancy_grid", showindex=False))
else:
    print("\n⚠️ No ANOVA results to display — all inputs were skipped or missing.")
