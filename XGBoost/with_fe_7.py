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

# Function to extract Deck from Cabin
def get_deck(cabin):
    if pd.isna(cabin):
        return "Unknown"
    else:
        return cabin[0]

# Add Deck feature to both datasets
train_data["Deck"] = train_data["Cabin"].apply(get_deck)
test_data["Deck"] = test_data["Cabin"].apply(get_deck)

# Separate target from features
y = train_data["Survived"]
X = train_data.drop(["Survived", "PassengerId", "Name", "Ticket", "Cabin", "Sex"], axis=1)
X_test = test_data.drop(["PassengerId", "Name", "Ticket", "Cabin", "Sex"], axis=1)

# Define preprocessing steps
numerical_features = ["Age", "SibSp", "Parch", "Fare"]
categorical_features = ["Pclass", "Embarked", "Deck"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), numerical_features),
        (
            "cat",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]
            ),
            categorical_features,
        ),
    ]
)

# Preprocess the data
X_processed = preprocessor.fit_transform(X)  # Fit and transform on training data
X_test_processed = preprocessor.transform(X_test)  # Transform only on test data

# Train XGBoost model
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
output.to_csv("submission_xgb_deck.csv", index=False)

# Feature importance
feature_names = (
    numerical_features + 
    preprocessor.named_transformers_["cat"]
    .named_steps["onehot"]
    .get_feature_names_out(categorical_features).tolist()
)
feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": model.feature_importances_
})
print(feature_importance.sort_values(by="Importance", ascending=False))