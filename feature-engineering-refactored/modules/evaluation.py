from sklearn.model_selection import cross_val_score


def evaluate_model(model, X, y, cv=5):
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    print(f"📈 CV Accuracy: \033[93m{scores.mean():.5f}\033[0m ± {scores.std():.5f}")
    return scores.mean()
