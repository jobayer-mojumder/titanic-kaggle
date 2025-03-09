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


# Function to create AgeGroup
def categorize_age(age):
    if pd.isna(age):
        return "Unknown"  # Handle missing values
    elif age <= 12:
        return "Child"
    elif age <= 60:
        return "Adult"
    else:
        return "Senior"


# Add AgeGroup to train and test data
train_data["AgeGroup"] = train_data["Age"].apply(categorize_age)
test_data["AgeGroup"] = test_data["Age"].apply(categorize_age)

# Separate target from features
y = train_data["Survived"]
X = train_data.drop(
    ["Survived", "PassengerId", "Name", "Ticket", "Cabin", "Sex", "Age"],
    axis=1,  # Drop Age since AgeGroup replaces it
)
X_test = test_data.drop(
    ["PassengerId", "Name", "Ticket", "Cabin", "Sex", "Age"],
    axis=1,  # Drop Age in test data too
)

# Define preprocessing steps
numerical_features = ["SibSp", "Parch", "Fare"]  # Removed Age
categorical_features = ["Pclass", "Embarked", "AgeGroup"]  # Added AgeGroup

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
output.to_csv("submission_xgb.csv", index=False)

# Optional: Feature importance
feature_names = (
    numerical_features
    + preprocessor.named_transformers_["cat"]
    .named_steps["onehot"]
    .get_feature_names_out(categorical_features)
    .tolist()
)
feature_importance = pd.DataFrame(
    {"Feature": feature_names, "Importance": model.feature_importances_}
)
print(feature_importance.sort_values(by="Importance", ascending=False))
