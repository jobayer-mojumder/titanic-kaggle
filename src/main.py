import pandas as pd
from modules.preprocessing import preprocess
from modules.feature_implementation import FEATURE_MAP
from modules.summary import log_results, get_submission_path
from modules.evaluation import evaluate_model
from modules.model_tuning import tune_model
from modules.constant import DEFAULT_MODELS
from modules.result_summary import run_best_results
from modules.combination_sampler import run_balanced_combinations
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

# Terminal color codes
BLUE = "\033[94m"
GREEN = "\033[92m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def run_model(model_key, feature_nums, use_cv=True, tune=False):
    selected_features = [FEATURE_MAP[n] for n in feature_nums]
    print(f"\n🚀 Running {model_key} with: {selected_features or 'Baseline only'}")

    train = pd.read_csv("../train.csv")
    test = pd.read_csv("../test.csv")

    y = train["Survived"]
    train.drop(columns=["Survived"], inplace=True)

    X_train, preproc = preprocess(train.copy(), selected_features, is_train=True)
    X_test, _ = preprocess(
        test.copy(), selected_features, is_train=False, ref_pipeline=preproc
    )

    if tune:
        model = tune_model(X_train, y, model_key=model_key)
    else:
        model = DEFAULT_MODELS[model_key]

    model.fit(X_train, y)
    preds = model.predict(X_test)

    acc, std = evaluate_model(model, X_train, y, model_name=model_key)

    out_file = get_submission_path(model_key, feature_nums)
    pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": preds}).to_csv(
        out_file, index=False
    )
    print(f"✅ Saved predictions to {out_file}")

    log_results(model_key, selected_features, acc, out_file, tuned=tune, std=std)


def run_all_general_combinations(model_key, tune=False):
    from modules.combination import GENERAL_FEATURE_COMBINATIONS

    for combo in GENERAL_FEATURE_COMBINATIONS:
        run_model(model_key, combo, use_cv=True, tune=tune)


def run_all_single_features(model_key, tune=False):
    for key in FEATURE_MAP.keys():
        run_model(model_key, [key], use_cv=True, tune=tune)
    run_model(model_key, [], use_cv=True, tune=tune)


def run_best_combinations(model_key, tune=False):
    from modules.combination import (
        DT_COMBINATIONS,
        XGB_COMBINATIONS,
        RF_COMBINATIONS,
        LGBM_COMBINATIONS,
        CB_COMBINATIONS,
    )

    model_combos = {
        "dt": DT_COMBINATIONS,
        "xgb": XGB_COMBINATIONS,
        "rf": RF_COMBINATIONS,
        "lgbm": LGBM_COMBINATIONS,
        "cb": CB_COMBINATIONS,
    }
    for combo in model_combos.get(model_key, []):
        run_model(model_key, combo, use_cv=True, tune=tune)


def run_all_models_single_features(tune=False):
    for model_key in ["dt", "xgb", "rf", "lgbm", "cb"]:
        run_all_single_features(model_key, tune=tune)


def run_all_models_general_combinations(tune=False):
    for model_key in ["dt", "xgb", "rf", "lgbm", "cb"]:
        run_all_general_combinations(model_key, tune=tune)


def run_all_models_best_combinations(tune=False):
    for model_key in ["dt", "xgb", "rf", "lgbm", "cb"]:
        run_best_combinations(model_key, tune=tune)


def print_menu():
    left_label = "Feature Engineering".ljust(31)
    right_label = "Model Tuning".ljust(31)

    print(f"\n{BOLD}{CYAN}🎯 Select an experiment to run...{RESET}\n")
    print("┌────────────────────────────────┬─────────────────────────────────┐")
    print(f"│ {BLUE}{left_label}{RESET}│ {GREEN}{right_label}{RESET} │")
    print("├────────────────────────────────┼─────────────────────────────────┤")
    print("│ 1. Single features             │ 7. Single features              │")
    print("│ 2. General combinations        │ 8. General combinations         │")
    print("│ 3. Best combinations           │ 9. Best combinations            │")
    print("│ 4. Feature Best Results        │ 10. Tuned Best Results          │")
    print("│ 5. Feature Combination Summary │ 11. Tuned Combination Summary   │")
    print("│ 6. All modes (no tuning)       │ 12. All modes (with tuning)     │")
    print("└────────────────────────────────┴─────────────────────────────────┘")
    print(" " * 30 + "0. Exit\n")

    print(f"{BOLD}Note: This will take a while!{RESET}\n")


def main():
    while True:
        print_menu()
        choice = input("Enter your choice (0–12): ").strip()

        if choice == "1":
            run_all_models_single_features(tune=False)
        elif choice == "2":
            run_all_models_general_combinations(tune=False)
        elif choice == "3":
            run_all_models_best_combinations(tune=False)
        elif choice == "4":
            run_best_results(tune=False)
        elif choice == "5":
            run_balanced_combinations(tune=False)
        elif choice == "6":
            run_all_models_single_features(tune=False)
            run_all_models_general_combinations(tune=False)
            run_all_models_best_combinations(tune=False)
        elif choice == "7":
            run_all_models_single_features(tune=True)
        elif choice == "8":
            run_all_models_general_combinations(tune=True)
        elif choice == "9":
            run_all_models_best_combinations(tune=True)
        elif choice == "10":
            run_best_results(tune=True)
        elif choice == "11":
            run_balanced_combinations(tune=True)
        elif choice == "12":
            run_all_models_single_features(tune=True)
            run_all_models_general_combinations(tune=True)
            run_all_models_best_combinations(tune=True)
        elif choice == "0":
            print("👋 Exiting.")
            break
        else:
            print("❌ Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
