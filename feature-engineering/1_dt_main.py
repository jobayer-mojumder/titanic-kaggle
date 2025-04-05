# type: ignore
import os
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from modules.feature_implementation import FEATURE_FUNCTIONS, FEATURE_MAP, SELECTED_FEATURES
from modules.combination import GENERAL_FEATURE_COMBINATIONS

# ------------------ Preprocessing ------------------

def preprocess(df, feature_names, is_train=True, reference_columns=None):
    # Apply feature engineering
    for f in feature_names:
        df = FEATURE_FUNCTIONS[f](df)

    # Always used baseline features
    if "Sex" in df.columns:
        df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    if "Pclass" in df.columns:
        df["Pclass"] = df["Pclass"].astype(str)

    # Drop unnecessary columns
    df = df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1, errors="ignore")

    # One-hot encode categoricals
    df = pd.get_dummies(df)

    # Align test set to train columns
    if not is_train:
        for col in reference_columns:
            if col not in df.columns:
                df[col] = 0
        df = df[reference_columns]

    return df

# ------------------ Runner ------------------

def run_dt(features_by_number):
    feature_names = [FEATURE_MAP[n] for n in features_by_number]
    global SELECTED_FEATURES
    SELECTED_FEATURES = feature_names

    print(f"🚀 Running with features: {feature_names or 'baseline only'}")

    train = pd.read_csv("../train.csv")
    test = pd.read_csv("../test.csv")

    y = train["Survived"]
    X = preprocess(train.drop("Survived", axis=1), feature_names, is_train=True)
    X_test = preprocess(test.copy(), feature_names, is_train=False, reference_columns=X.columns)

    # Impute missing values
    imputer = SimpleImputer(strategy="median")
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    X_test = pd.DataFrame(imputer.transform(X_test), columns=X.columns)

    # Train Decision Tree
    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    model.fit(X, y)
    preds = model.predict(X_test)

    # Save predictions
    os.makedirs("submissions/1_Decision-Tree", exist_ok=True)
    suffix = "base" if not features_by_number else "_".join(map(str, features_by_number))
    filename = f"submissions/1_Decision-Tree/submission_dt_{suffix}.csv"

    pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": preds}).to_csv(filename, index=False)
    print(f"✅ Saved to {filename}")


def run_all_single_features():
    for i in range(1, len(FEATURE_MAP) + 1):
        run_dt([i])
    run_dt([])

def run_general_combinations():
    # Run all general combinations
    for combination in GENERAL_FEATURE_COMBINATIONS:
        run_dt(combination)

# ------------------ Main ------------------

if __name__ == "__main__":
    # run_all_single_features()
    run_general_combinations()
