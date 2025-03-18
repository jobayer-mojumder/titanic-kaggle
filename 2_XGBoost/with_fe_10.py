# type: ignore
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb

# Load datasets
train_data = pd.read_csv("../train.csv")
test_data = pd.read_csv("../test.csv")

# Add IsChild feature
train_data['IsChild'] = (train_data['Age'].between(0, 12)).astype(int)
test_data['IsChild'] = (test_data['Age'].between(0, 12)).astype(int)

# Separate target from features
y = train_data["Survived"]
X = train_data.drop(
    ["Survived", "PassengerId", "Name", "Ticket", "Cabin"], axis=1
)
X_test = test_data.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)

# Convert Sex to binary (1 for female, 0 for male)
X["Sex"] = X["Sex"].map({"female": 1, "male": 0})
X_test["Sex"] = X_test["Sex"].map({"female": 1, "male": 0})

# Define preprocessing steps
numerical_features = ["Age", "SibSp", "Parch", "Fare", "IsChild", "Sex"]  # Add IsChild
categorical_features = ["Pclass", "Embarked"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), numerical_features),
        (
            "cat",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder()),
                ]
            ),
            categorical_features,
        ),
    ]
)

# Preprocess the data
X_processed = preprocessor.fit_transform(X)
X_test_processed = preprocessor.transform(X_test)

# Initialize and train XGBoost model
model = xgb.XGBClassifier(
    use_label_encoder=False, eval_metric="logloss", random_state=42
)
model.fit(X_processed, y)

# Generate predictions for test set
predictions = model.predict(X_test_processed)

# Create submission file
output = pd.DataFrame(
    {"PassengerId": test_data["PassengerId"], "Survived": predictions}
)
output.to_csv("submission_xgb_ischild.csv", index=False)