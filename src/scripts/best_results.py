# best_results_safe.py
# type: ignore
import pandas as pd
import os
import sys

# --- CLI argument handling ---
if len(sys.argv) > 1:
    arg = sys.argv[1].lower()
    if arg in ("--help", "-h"):
        print("Usage: python best_results_safe.py [tune]")
        print("Pass 'true' or '1' to enable tuning mode.")
        sys.exit(0)
    tune = arg in ("true", "1")
else:
    tune = False

print(f"🔍 Mode: {'TUNING' if tune else 'NORMAL'}")

# --- Model file map ---
file_paths = {
    "Decision Tree": "1_dt_comb.csv",
    "XGBoost": "2_xgb_comb.csv",
    "Random Forest": "3_rf_comb.csv",
    "LightGBM": "4_lgbm_comb.csv",
    "CatBoost": "5_cb_comb.csv",
}

best_results = []

for model, path in file_paths.items():
    read_path = os.path.join(
        "../combination-results/tuning" if tune else "../combination-results",
        path.replace("_comb", "_tune_comb") if tune else path,
    )

    if not os.path.exists(read_path):
        print(f"⚠️ File not found: {read_path}")
        continue

    try:
        df = pd.read_csv(read_path)

        if df.empty:
            print(f"⚠️ File is empty: {read_path}")
            continue

        if "accuracy_vs_kaggle" not in df.columns:
            print(f"❌ Column 'accuracy_vs_kaggle' missing in {read_path}")
            continue

        best_row = df.loc[df["accuracy_vs_kaggle"].idxmax()].copy()
        best_row["model_name"] = model
        best_results.append(best_row)

    except pd.errors.EmptyDataError:
        print(f"⚠️ No data in file: {read_path}")
    except Exception as e:
        print(f"❌ Error processing {read_path}: {e}")

# --- Final output ---
if best_results:
    best_df = pd.DataFrame(best_results)
    print("\n✅ Best results per model:")
    print(best_df)

    out_path = "../results/best_results.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    best_df.to_csv(out_path, index=False)
    print(f"\n💾 Saved to {out_path}")
else:
    print("\n⚠️ No valid results found to save.")
