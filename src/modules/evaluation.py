# type: ignore
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedShuffleSplit


def evaluate_model(model, X, y, cv=5, model_name="", random_state=42):

    skf = StratifiedShuffleSplit(n_splits=cv, test_size=0.2, random_state=random_state)
    scores = cross_val_score(model, X, y, cv=skf, scoring="accuracy")

    acc = round(scores.mean(), 5)
    std = round(scores.std(), 5)

    scores_str = "[" + ", ".join(f"{s:.8f}" for s in scores) + "]"

    print(f"📈 CV Accuracy: \033[93m{acc}\033[0m ± {std:.5f}")

    return acc, std, scores_str
