# type: ignore
import pandas as pd
from lightgbm import LGBMClassifier
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
        ["PassengerId", "Name", "Ticket", "Cabin", "FamilySize"],
        axis=1,  # Drop FamilySize after use
    )
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    df = pd.get_dummies(df, columns=["Embarked", "Pclass"])

    # Impute missing values
    imputer = SimpleImputer(strategy="median")
    return pd.DataFrame(imputer.fit_transform(df), columns=df.columns)


# Prepare data
X_train = preprocess(train.drop("Survived", axis=1))
y_train = train["Survived"]
X_test = preprocess(test)

# Train LightGBM model
model = LGBMClassifier(n_estimators=100, max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Create submission
pd.DataFrame(
    {"PassengerId": test["PassengerId"], "Survived": model.predict(X_test)}
).to_csv("submission_lgbm_fareperperson.csv", index=False)

# Optional: Feature importance
feature_importance = pd.DataFrame(
    {"Feature": X_train.columns, "Importance": model.feature_importances_}
)
print(feature_importance.sort_values(by="Importance", ascending=False))
