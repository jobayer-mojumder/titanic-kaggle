# type: ignore
from sklearn.model_selection import cross_val_score
from modules.constant import BASELINE_SCORE
from sklearn.model_selection import StratifiedKFold


def evaluate_model(model, X, y, cv=10, model_name=""):

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    scores = cross_val_score(model, X, y, cv=skf, scoring="accuracy")

    acc = round(scores.mean(), 5)
    std = round(scores.std(), 5)

    scores_str = "[" + ", ".join(f"{s:.8f}" for s in scores) + "]"

    print(f"📈 CV Accuracy: \033[93m{acc}\033[0m ± {std:.5f}")

    return acc, std, scores_str
