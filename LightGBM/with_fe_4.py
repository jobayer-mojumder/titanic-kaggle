# type: ignore
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
train = pd.read_csv("../train.csv")
test = pd.read_csv("../test.csv")


def create_is_alone(df):
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1  # Create FamilySize first
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)  # 1 if alone, else 0
    df = df.drop(["FamilySize"], axis=1)  # Drop FamilySize if not needed
    return df


# Apply feature engineering to both train and test data
train = create_is_alone(train)
test = create_is_alone(test)


# Same preprocessing as previous models
def preprocess(df):
    df = df.drop(
        [
            "PassengerId",
            "Name",
            "Ticket",
            "Cabin",
            "Sex",
        ],
        axis=1,
    )
    df = pd.get_dummies(df, columns=["Embarked", "Pclass"])

    # Impute missing values (same strategy)
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
).to_csv("lgbm_fe_4.csv", index=False)
