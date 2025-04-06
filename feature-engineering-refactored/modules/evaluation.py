# type: ignore
from sklearn.model_selection import cross_val_score
from modules.constant import BASELINE_SCORE


def evaluate_model(model, X, y, cv=5, model_name=""):
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")

    acc = scores.mean()
    acc = round(acc, 5)

    print(f"📈 CV Accuracy: \033[93m{acc}\033[0m ± {scores.std():.5f}")

    if model_name:
        baseline_score = round(BASELINE_SCORE[model_name], 5)

        if acc == baseline_score:
            print(f"\033[93m⚠️  Accuracy matches baseline: {acc} \033[0m")
        elif acc > baseline_score:
            print(
                f"\033[92m✅ Accuracy improved: {acc} vs baseline {baseline_score} = +{(acc - baseline_score):.5f} \033[0m"
            )
        else:
            print(
                f"\033[91m⚠️  Accuracy drop detected: {acc} vs baseline {baseline_score} = -{(baseline_score - acc):.5f} \033[0m"
            )

    return acc
