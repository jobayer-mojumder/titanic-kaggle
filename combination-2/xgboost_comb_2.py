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

# Add IsMother feature
train_data["IsMother"] = ((train_data["Sex"] == "female") & (train_data["Parch"] > 0) & (train_data["Age"] > 18)).astype(int)
test_data["IsMother"] = ((test_data["Sex"] == "female") & (test_data["Parch"] > 0) & (test_data["Age"] > 18)).astype(int)

# Convert Sex to binary (1 for female, 0 for male)
train_data['Sex'] = train_data['Sex'].map({'female': 1, 'male': 0})
test_data['Sex'] = test_data['Sex'].map({'female': 1, 'male': 0})

# Create SexPclass feature
train_data['SexPclass'] = train_data['Sex'] * train_data['Pclass']
test_data['SexPclass'] = test_data['Sex'] * test_data['Pclass']

# Separate target from features and drop Sex explicitly
y = train_data["Survived"]
X = train_data.drop(
    ["Survived", "PassengerId", "Name", "Ticket", "Cabin"], axis=1
)
X_test = test_data.drop(
    ["PassengerId", "Name", "Ticket", "Cabin"], axis=1
)

# Define preprocessing steps
numerical_features = ["Age", "SibSp", "Parch", "Fare", "SexPclass", "Sex", "WomenChildrenFirst", "IsMother"]
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
output.to_csv("submission_xgb_comb_2.csv", index=False)