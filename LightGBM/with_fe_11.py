# type: ignore
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
train = pd.read_csv("../train.csv")
test = pd.read_csv("../test.csv")

# Add IsChild feature
train['IsChild'] = (train['Age'].between(0, 12)).astype(int)
test['IsChild'] = (test['Age'].between(0, 12)).astype(int)

# Same preprocessing as previous models
def preprocess(df):
    df = df.drop(
        ["PassengerId", "Name", "Ticket", "Cabin", "Sex"],
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
).to_csv("submission_lgbm_ischild.csv", index=False)