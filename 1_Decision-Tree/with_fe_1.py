# type: ignore
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load data
train = pd.read_csv("../train.csv")
test = pd.read_csv("../test.csv")


# Basic preprocessing
def preprocess(df):
    df["Title"] = df["Name"].str.extract(" ([A-Za-z]+)\.", expand=False)
    df["Title"] = df["Title"].replace(
        [
            "Lady",
            "Countess",
            "Capt",
            "Col",
            "Don",
            "Dr",
            "Major",
            "Rev",
            "Sir",
            "Jonkheer",
            "Dona",
        ],
        "Rare",
    )

    df["Title"] = df["Title"].replace(["Mlle", "Ms"], "Miss")
    df["Title"] = df["Title"].replace(["Mme"], "Mrs")

    df = df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)

    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

    df = pd.get_dummies(df, columns=["Embarked", "Pclass", "Title"])

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
    "submission_dt_title.csv", index=False
)
