# type: ignore
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

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
X = preprocess(train.drop("Survived", axis=1))
y = train["Survived"]

# Train model
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X, y)

# Prepare test predictions
X_test = preprocess(test)
test_predictions = model.predict(X_test)

# Create submission
pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": test_predictions}).to_csv(
    "submission_dt_fareperperson.csv", index=False
)

# Optional: Feature importance
feature_importance = pd.DataFrame(
    {"Feature": X.columns, "Importance": model.feature_importances_}
)
print(feature_importance.sort_values(by="Importance", ascending=False))
