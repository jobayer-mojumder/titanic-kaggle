# type: ignore
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
train = pd.read_csv("../train.csv")
test = pd.read_csv("../test.csv")


# Updated preprocessing function to include FarePerPerson
def preprocess(df):
    # Calculate FamilySize and FarePerPerson
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["FarePerPerson"] = df["Fare"] / df["FamilySize"]

    # Drop unnecessary columns
    df = df.drop(
        ["PassengerId", "Name", "Ticket", "Cabin", "Sex", "FamilySize"], axis=1
    )  # Drop FamilySize after use
    df = pd.get_dummies(df, columns=["Embarked", "Pclass"])

    # Impute missing values
    imputer = SimpleImputer(strategy="median")
    return pd.DataFrame(imputer.fit_transform(df), columns=df.columns)


# Prepare data
X_train = preprocess(train.drop("Survived", axis=1))
y_train = train["Survived"]
X_test = preprocess(test)

# Train CatBoost model
model = CatBoostClassifier(
    iterations=100, depth=3, random_seed=42, verbose=0  # Silent mode
)
model.fit(X_train, y_train)

# Create submission
pd.DataFrame(
    {"PassengerId": test["PassengerId"], "Survived": model.predict(X_test)}
).to_csv("submission_catboost_fareperperson.csv", index=False)

# Optional: Feature importance
feature_importance = pd.DataFrame(
    {"Feature": X_train.columns, "Importance": model.get_feature_importance()}
)
print(feature_importance.sort_values(by="Importance", ascending=False))
