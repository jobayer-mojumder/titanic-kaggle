# type: ignore
import os
from handlers.dispatcher import print_menu, handle_choice
from modules.feature_implementation import FEATURE_MAP
from modules.preprocessing import preprocess
from modules.evaluation import evaluate_model
from modules.summary import log_results, get_submission_path
from modules.model_tuning import tune_model
from modules.constant import DEFAULT_MODELS
import pandas as pd
import warnings

warnings.filterwarnings("ignore")


def run_model(model_key, feature_nums, tune=False):
    selected_features = [FEATURE_MAP[n] for n in feature_nums]
    print(f"\n🚀 Running {model_key} with: {selected_features or 'Baseline only'}")

    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train["Survived"]
    train.drop(columns=["Survived"], inplace=True)

    X_train, preproc = preprocess(train.copy(), selected_features, is_train=True)
    X_test, _ = preprocess(
        test.copy(), selected_features, is_train=False, ref_pipeline=preproc
    )

    if tune:
        model, best_params = tune_model(X_train, y, model_key=model_key)
    else:
        model = DEFAULT_MODELS[model_key]
        best_params = None

    model.fit(X_train, y)
    preds = model.predict(X_test)

    acc, std = evaluate_model(model, X_train, y, model_name=model_key)
    out_file = get_submission_path(model_key, feature_nums)
    pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": preds}).to_csv(
        out_file, index=False
    )
    print(f"✅ Saved predictions to {os.path.basename(out_file)}")
    log_results(
        model_key,
        selected_features,
        acc,
        out_file,
        tuned=tune,
        params=best_params,
        std=std,
    )


def main():
    last_activity = None
    last_duration = None

    while True:
        os.system("cls" if os.name == "nt" else "clear")
        print_menu(last_activity, last_duration)
        choice = input("Enter your choice (0–14): ").strip()
        if choice == "0":
            print("👋 Exiting.")
            break
        last_activity, last_duration = handle_choice(choice, run_model)


if __name__ == "__main__":
    main()
