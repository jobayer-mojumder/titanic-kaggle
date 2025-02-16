# type: ignore
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb
from sklearn.metrics import accuracy_score

# Load datasets
train_data = pd.read_csv("../train.csv")
test_data = pd.read_csv("../test.csv")


def extract_title(df):
    df["Title"] = df["Name"].str.extract(" ([A-Za-z]+)\.", expand=False)
    # Replace rare titles with more common groups
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
    # Group similar titles
    df["Title"] = df["Title"].replace(["Mlle", "Ms"], "Miss")
    df["Title"] = df["Title"].replace(["Mme"], "Mrs")
    return df


# Apply title extraction to both train and test data
train_data = extract_title(train_data)
test_data = extract_title(test_data)

# Separate target from features
y = train_data["Survived"]
X = train_data.drop(
    ["Survived", "PassengerId", "Name", "Ticket", "Cabin", "Sex"], axis=1
)
X_test = test_data.drop(["PassengerId", "Name", "Ticket", "Cabin", "Sex"], axis=1)

# Define preprocessing steps
numerical_features = ["Age", "SibSp", "Parch", "Fare"]
categorical_features = ["Pclass", "Embarked", "Title"]

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
output.to_csv("xgb_fe_1.csv", index=False)
