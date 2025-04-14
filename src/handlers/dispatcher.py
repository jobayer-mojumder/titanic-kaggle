from handlers.runner import (
    run_all_models,
    run_all_single_features,
    run_all_general_combinations,
    run_model_combinations,
    run_baseline_models,
)
from handlers.viewer import handle_viewer
from handlers.utils import prompt_all_or_one, select_model_key
from modules.result_summary import run_best_results
from modules.combination_sampler import run_balanced_combinations
import time

BLUE = "\033[94m"
GREEN = "\033[92m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def handle_choice(choice, run_model):
    start = time.time()

    def route(label, batch_fn, one_fn, tune):
        all_or_one = prompt_all_or_one()
        if all_or_one == "o":
            model = select_model_key()
            if model:
                one_fn(run_model, model, tune=tune)
                return (
                    f"{label} for {model} ({'tuning' if tune else 'no tuning'})",
                    time.time() - start,
                )
            else:
                print("❌ Invalid model.")
                input("🔁 Press Enter to return...")
                return None, None
        else:
            run_all_models(run_model, batch_fn, tune=tune)
            return (
                f"{label} for all models ({'tuning' if tune else 'no tuning'})",
                time.time() - start,
            )

    if choice == "1":
        return route(
            "Single features", run_all_single_features, run_all_single_features, False
        )
    elif choice == "2":
        return route(
            "General combinations",
            run_all_general_combinations,
            run_all_general_combinations,
            False,
        )
    elif choice == "3":
        return route(
            "Model combinations", run_model_combinations, run_model_combinations, False
        )
    elif choice == "4":
        handle_viewer("Feature Best Results", lambda: run_best_results(tune=False))
        return "Viewed feature best results", time.time() - start
    elif choice == "5":
        handle_viewer(
            "Feature Combination Summary", lambda: run_balanced_combinations(tune=False)
        )
        return "Viewed feature combination summary", time.time() - start
    elif choice == "6":
        for fn in [
            run_all_single_features,
            run_all_general_combinations,
            run_model_combinations,
        ]:
            run_all_models(run_model, fn, tune=False)
        return "All modes (no tuning)", time.time() - start
    elif choice == "7":
        return route(
            "Single features", run_all_single_features, run_all_single_features, True
        )
    elif choice == "8":
        return route(
            "General combinations",
            run_all_general_combinations,
            run_all_general_combinations,
            True,
        )
    elif choice == "9":
        return route(
            "Model combinations", run_model_combinations, run_model_combinations, True
        )
    elif choice == "10":
        handle_viewer("Tuned Best Results", lambda: run_best_results(tune=True))
        return "Viewed tuned best results", time.time() - start
    elif choice == "11":
        handle_viewer(
            "Tuned Combination Summary", lambda: run_balanced_combinations(tune=True)
        )
        return "Viewed tuned combination summary", time.time() - start
    elif choice == "12":
        for fn in [
            run_all_single_features,
            run_all_general_combinations,
            run_model_combinations,
        ]:
            run_all_models(run_model, fn, tune=True)
        return "All modes (tuning)", time.time() - start
    elif choice == "13":
        return route("Baseline model", run_baseline_models, run_baseline_models, False)
    elif choice == "14":
        return route("Baseline model", run_baseline_models, run_baseline_models, True)

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
            print(f"{BLUE}⏱️ Time taken:{RESET} {format_duration(last_duration)}")
    print("\n┌────────────────────────────────┬─────────────────────────────────┐")
    print(
        f"│ {BLUE}Feature Engineering{RESET}            │ {GREEN}Model Tuning{RESET}                    │"
    )
    print("├────────────────────────────────┼─────────────────────────────────┤")
    print("│ 1. Single features             │ 7. Single features              │")
    print("│ 2. General combinations        │ 8. General combinations         │")
    print("│ 3. Model combinations          │ 9. Model combinations           │")
    print("│ 4. Feature Best Results        │ 10. Tuned Best Results          │")
    print("│ 5. Feature Combination Summary │ 11. Tuned Combination Summary   │")
    print("│ 6. All modes (no tuning)       │ 12. All modes (with tuning)     │")
    print("│ 13. Run baseline model         │ 14. Run baseline model (tuned)  │")
    print("└────────────────────────────────┴─────────────────────────────────┘")
    print(" " * 30 + "0. Exit\n")
