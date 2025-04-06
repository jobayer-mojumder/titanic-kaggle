# type: ignore
import pandas as pd
import os


def log_results(model_name, feature_list, accuracy, output_file="results_summary.csv"):
    if not feature_list:
        mode = "baseline"
    elif len(feature_list) == 1:
        mode = "single"
    else:
        mode = "combo"

    row = {
        "model": model_name,
        "features": ", ".join(feature_list) if feature_list else "baseline",
        "accuracy": accuracy,
        "mode": mode,
    }

    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        df = pd.read_csv(output_file)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_csv(output_file, index=False)
    print(f"📊 Logged result for {model_name} | Mode: {mode}")
