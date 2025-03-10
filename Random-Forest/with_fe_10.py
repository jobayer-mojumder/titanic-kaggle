# type: ignore
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
train = pd.read_csv("../train.csv")
test = pd.read_csv("../test.csv")

# Add SexPclass feature before preprocessing
train['Sex'] = train['Sex'].map({'female': 1, 'male': 0})
test['Sex'] = test['Sex'].map({'female': 1, 'male': 0})
train['SexPclass'] = train['Sex'] * train['Pclass']
test['SexPclass'] = test['Sex'] * test['Pclass']

# Basic preprocessing
def preprocess(df):
    df = df.drop(["PassengerId", "Name", "Ticket", "Cabin", "Sex"], axis=1)  # Drop Sex after using it
    df = pd.get_dummies(df, columns=["Embarked", "Pclass"])

    # Impute missing values
    imputer = SimpleImputer(strategy="median")
    return pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Prepare data
X = preprocess(train.drop("Survived", axis=1))
y = train["Survived"]
X_test = preprocess(test)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Create submission
pd.DataFrame(
    {"PassengerId": test["PassengerId"], "Survived": model.predict(X_test)}
).to_csv("submission_rf_sex_pclass.csv", index=False)