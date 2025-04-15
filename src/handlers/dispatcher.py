from handlers.runner import (
    run_all_models,
    run_all_single_features,
    run_all_general_combinations,
    run_baseline_models,
    run_all_features_in_one_combination,
    run_best_single_feature_combination,
    run_10_balanced_feature_combinations,
    run_10_best_feature_combinations,
)
import time

BLUE = "\033[94m"
GREEN = "\033[92m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def handle_choice(choice, run_model):
    start = time.time()

    def route(label, batch_fn, tune):
        run_all_models(run_model, batch_fn, tune=tune)
        return (
            f"{label} for all models ({'tuning' if tune else 'no tuning'})",
            time.time() - start,
        )

    if choice == "1":
        return route("Baseline models", run_baseline_models, False)
    elif choice == "2":
        return route(
            "Single features (FE) for all models",
            run_all_single_features,
            False,
        )
    elif choice == "3":
        return route(
            "General combinations (FE) for all models",
            run_all_general_combinations,
            False,
        )
    elif choice == "4":
        return route(
            "All features in one combinations (FE) for all models",
            run_all_features_in_one_combination,
            False,
        )
    elif choice == "5":
        return route("Baseline models", run_baseline_models, True)
    elif choice == "6":
        return route(
            "Single features (MT) for all models",
            run_all_single_features,
            True,
        )
    elif choice == "7":
        return route(
            "All features in one combinations (MT) for all models",
            run_all_features_in_one_combination,
            True,
        )
    elif choice == "8":
        return route(
            "Best single FE combinations (MT) for all models",
            run_best_single_feature_combination,
            True,
        )
    elif choice == "9":
        return route(
            "10 Balanced FE combinations (MT) for all models",
            run_10_balanced_feature_combinations,
            True,
        )
    elif choice == "10":
        return route(
            "10 Best FE combinations (MT) for all models",
            run_10_best_feature_combinations,
            True,
        )
    else:
        print("Invalid choice. Please try again.")

    return None, None


def format_duration(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours}h {minutes}m {secs:.2f}s"


def print_menu(last_activity=None, last_duration=None):
    print(f"\n{BOLD}{CYAN}🎯 Select an experiment to run:{RESET}\n")
    if last_activity:
        print(f"{GREEN}🕘 Last activity:{RESET} {last_activity}")
        if last_duration is not None:
            print(f"{BLUE}⏱️  Time taken:{RESET} {format_duration(last_duration)}")

    print("\n┌──────────────────────────────────────────┐")
    print("│ 1. Baseline model (FE)                   │")
    print("│ 2. Single features (FE)                  │")
    print("│ 3. General combinations (FE)             │")
    print("│ 4. All features in one combinations (FE) │")
    print("│ 5. Baseline model (MT)                   │")
    print("│ 6. Single features (MT)                  │")
    print("│ 7. All features in one combinations (MT) │")
    print("│ 8. Best single FE combinations (MT)      │")
    print("│ 9. 10 Balanced FE combinations (MT)      │")
    print("│ 10. 10 Best FE combinations (MT)         │")
    print("└──────────────────────────────────────────┘")
    print("\n0. Exit\n")
