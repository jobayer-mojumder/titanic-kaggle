import pandas as pd
import ast
import os


def generate_jasp_ready_data(
    input_path, output_folder, expand_variance=False, top_k=5, distinct=True
):
    summary = pd.read_csv(input_path)

    # Clean and convert columns
    summary["feature_num"] = summary["feature_num"].astype(str)
    summary["tuned"] = summary["tuned"].fillna(0).astype(int)
    summary["model"] = summary["model"].astype(str)
    summary["accuracy"] = summary["accuracy"].fillna(0)
    summary["cv_scores"] = summary["cv_scores"].fillna("[]")

    balanced_rows = []

    for model in summary["model"].unique():
        model_data = summary[summary["model"] == model]

        # 1. Baseline: tuned = 0, feature_num = baseline (use CV scores)
        baseline_rows = model_data[
            (model_data["tuned"] == 0) & (model_data["feature_num"] == "baseline")
        ]
        for _, row in baseline_rows.iterrows():
            scores = safe_parse_cv(row["cv_scores"])
            for score in scores[:top_k]:
                balanced_rows.append(
                    {
                        "accuracy": score,
                        "Feature_Engineering": 0,
                        "Model_Tuning": 0,
                        "model": model,
                    }
                )

        # 2. Feature Engineering: tuned = 0, feature_num ≠ baseline
        fe_rows = model_data[
            (model_data["tuned"] == 0) & (model_data["feature_num"] != "baseline")
        ].sort_values(by="accuracy", ascending=False)
        if distinct:
            fe_rows = fe_rows.drop_duplicates(subset=["accuracy"])
        for _, row in fe_rows.head(top_k).iterrows():
            balanced_rows.append(
                {
                    "accuracy": row["accuracy"],
                    "Feature_Engineering": 1,
                    "Model_Tuning": 0,
                    "model": model,
                }
            )

        # 3. Model Tuning: tuned = 1, feature_num = baseline (use CV scores)
        mt_rows = model_data[
            (model_data["tuned"] == 1) & (model_data["feature_num"] == "baseline")
        ]
        for _, row in mt_rows.iterrows():
            scores = safe_parse_cv(row["cv_scores"])
            for score in scores[:top_k]:
                balanced_rows.append(
                    {
                        "accuracy": score,
                        "Feature_Engineering": 0,
                        "Model_Tuning": 1,
                        "model": model,
                    }
                )

        # 4. FE + MT: tuned = 1, feature_num ≠ baseline
        femt_rows = model_data[
            (model_data["tuned"] == 1) & (model_data["feature_num"] != "baseline")
        ].sort_values(by="accuracy", ascending=False)
        if distinct:
            femt_rows = femt_rows.drop_duplicates(subset=["accuracy"])
        for _, row in femt_rows.head(top_k).iterrows():
            balanced_rows.append(
                {
                    "accuracy": row["accuracy"],
                    "Feature_Engineering": 1,
                    "Model_Tuning": 1,
                    "model": model,
                }
            )

    df_out = pd.DataFrame(balanced_rows)
    os.makedirs(output_folder, exist_ok=True)
    filename = "anova_local_balanced.csv"
    output_path = os.path.join(output_folder, filename)
    df_out.to_csv(output_path, index=False)

    print(f"✅ Saved balanced ANOVA-ready data to {output_path}")


def safe_parse_cv(cv_str):
    try:
        scores = ast.literal_eval(cv_str)
        if isinstance(scores, list):
            return scores
    except Exception:
        pass
    return []
