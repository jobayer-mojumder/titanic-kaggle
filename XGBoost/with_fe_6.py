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

# Add FarePerPerson to train and test data
train_data["FamilySize"] = train_data["SibSp"] + train_data["Parch"] + 1
train_data["FarePerPerson"] = train_data["Fare"] / train_data["FamilySize"]
test_data["FamilySize"] = test_data["SibSp"] + test_data["Parch"] + 1
test_data["FarePerPerson"] = test_data["Fare"] / test_data["FamilySize"]

# Separate target from features
y = train_data["Survived"]
X = train_data.drop(
    ["Survived", "PassengerId", "Name", "Ticket", "Cabin", "Sex", "FamilySize"],
    axis=1,  # Drop FamilySize after use
)
X_test = test_data.drop(
    ["PassengerId", "Name", "Ticket", "Cabin", "Sex", "FamilySize"],
    axis=1,  # Drop FamilySize after use
)

# Define preprocessing steps
numerical_features = [
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "FarePerPerson",
]  # Added FarePerPerson
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
output.to_csv("submission_xgb_fareperperson.csv", index=False)

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

# Optional: Plot feature importance
import matplotlib.pyplot as plt

feature_importance.sort_values(by="Importance", ascending=False).plot.barh(
    x="Feature", y="Importance"
)
plt.show()
