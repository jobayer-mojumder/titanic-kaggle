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

# Add WomenChildrenFirst feature
train_data['WomenChildrenFirst'] = ((train_data['Sex'] == 'female') | (train_data['Age'] <= 12)).astype(int)
test_data['WomenChildrenFirst'] = ((test_data['Sex'] == 'female') | (test_data['Age'] <= 12)).astype(int)

# Separate target from features
y = train_data["Survived"]
X = train_data.drop(
    ["Survived", "PassengerId", "Name", "Ticket", "Cabin", "Sex"], axis=1
)
X_test = test_data.drop(["PassengerId", "Name", "Ticket", "Cabin", "Sex"], axis=1)

# Define preprocessing steps
numerical_features = ["Age", "SibSp", "Parch", "Fare", "WomenChildrenFirst"]  # Add WomenChildrenFirst
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
output.to_csv("submission_xgb_womenchildrenfirst.csv", index=False)