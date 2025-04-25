import pandas as pd
import os
import ast


def save_csv(df, filename, mode="local"):
    out_dir = f"jasp_outputs/{mode}"
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    df.to_csv(path, index=False)
    print(f"✅ Saved: {path}")


def extract_cv_scores(row):
    if isinstance(row, str):
        try:
            return ast.literal_eval(row)
        except Exception:
            return []
    return []


def get_baseline_scores(df_model, score_col, tuned, cv_col=None):
    baseline_row = df_model[
        (df_model["feature_num"] == "baseline") & (df_model["tuned"] == tuned)
    ]
    if baseline_row.empty:
        return []

    if cv_col:
        scores = []
        for _, row in baseline_row.iterrows():
            scores += extract_cv_scores(row.get(cv_col, ""))
        return scores
    else:
        return baseline_row[score_col].dropna().tolist()


def get_group_scores(df_model, score_col, tuned, group_name, cv_col=None):
    df = df_model[
        (df_model["feature_num"] != "baseline") & (df_model["tuned"] == tuned)
    ]
    scores = df[score_col].dropna().tolist()
    return pd.DataFrame({"group": group_name, "score": scores})


def export_comparison(df_model, model, score_col, cv_col, mode):
    baseline_scores = get_baseline_scores(df_model, score_col, tuned=0, cv_col=cv_col)
    if not baseline_scores:
        print(f"⚠️ Skipping {model.upper()} — no untuned baseline")
        return

    baseline_df = pd.DataFrame({"group": "baseline", "score": baseline_scores})
    baseline_df["model"] = model

    comparisons = [
        {
            "label": "feature_eng",
            "tuned": 0,
            "filename": f"{mode}_{model.lower()}_fe_vs_baseline.csv",
        },
        {
            "label": "tuning",
            "tuned": 1,
            "filename": f"{mode}_{model.lower()}_tuning_vs_baseline.csv",
        },
        {
            "label": "fe_tuning",
            "tuned": 1,
            "filename": f"{mode}_{model.lower()}_fe_mt_vs_baseline.csv",
        },
    ]

    for comp in comparisons:
        group_df = get_group_scores(
            df_model, score_col, tuned=comp["tuned"], group_name=comp["label"]
        )
        if group_df.empty:
            print(f"⚠️ Skipping {model.upper()} - {comp['label']} - no data")
            continue

        group_df["model"] = model
        combined = pd.concat([baseline_df, group_df], ignore_index=True)
        save_csv(combined[["model", "group", "score"]], comp["filename"], mode=mode)


def prepare_and_save_all_comparisons(summary_df, mode="local"):
    score_col = "accuracy" if mode == "local" else "kaggle_score"
    cv_col = "cv_scores" if mode == "local" else None

    for model in summary_df["model"].unique():
        print(f"\n🔍 Processing {model.upper()} ({mode})")
        df_model = summary_df[summary_df["model"] == model].copy()
        export_comparison(df_model, model, score_col, cv_col, mode)


def create_combined_fe_vs_baseline(summary_df, mode="local"):
    score_col = "accuracy" if mode == "local" else "kaggle_score"
    cv_col = "cv_scores" if mode == "local" else None

    all_rows = []

    for model in summary_df["model"].unique():
        df_model = summary_df[summary_df["model"] == model].copy()

        # ⬅️ Baseline CV scores
        baseline_scores = get_baseline_scores(
            df_model, score_col, tuned=0, cv_col=cv_col
        )
        for score in baseline_scores:
            all_rows.append({"model": model, "group": "baseline", "score": score})

        # ⬅️ Feature Engineering scores (untuned)
        fe_df = df_model[
            (df_model["feature_num"] != "baseline") & (df_model["tuned"] == 0)
        ]
        for _, row in fe_df.iterrows():
            all_rows.append(
                {"model": model, "group": "feature_eng", "score": row[score_col]}
            )

    if all_rows:
        out_df = pd.DataFrame(all_rows)
        out_path = f"jasp_outputs/{mode}/all_models_fe_vs_baseline.csv"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        out_df.to_csv(out_path, index=False)
        print(f"✅ Saved: {out_path}")
    else:
        print(f"⚠️ No data to save for FE vs Baseline ({mode})")


# 🚀 Run both modes
prepare_and_save_all_comparisons(pd.read_csv("results/summary_local.csv"), mode="local")
create_combined_fe_vs_baseline(pd.read_csv("results/summary_local.csv"), mode="local")

prepare_and_save_all_comparisons(
    pd.read_csv("results/summary_kaggle.csv"), mode="kaggle"
)
create_combined_fe_vs_baseline(pd.read_csv("results/summary_kaggle.csv"), mode="kaggle")
