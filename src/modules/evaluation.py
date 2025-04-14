# type: ignore
from sklearn.model_selection import cross_val_score
from modules.constant import BASELINE_SCORE
from modules.summary import compare_with_baseline


def evaluate_model(model, X, y, cv=5, model_name=""):
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    acc = round(scores.mean(), 5)
    std = round(scores.std(), 5)

    print(f"📈 CV Accuracy: \033[93m{acc}\033[0m ± {std:.5f}")

    # if model_name and model_name in BASELINE_SCORE:
    #     compare_with_baseline(acc, BASELINE_SCORE[model_name], label="CV Accuracy")

    return acc, std
