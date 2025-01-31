# Import libraries
# type: ignore
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Explore the data
# print(train_data.head())
# print(train_data.info())
# print(train_data.describe())

# Check for missing values
# print("Missing values in train data:\n")
# print(train_data.isnull().sum())

# Fill missing values
train_data['Age'].fillna(train_data['Age'].median(), inplace=True)
train_data['Embarked'].fillna(train_data['Embarked'].mode()[0], inplace=True)
train_data.drop('Cabin', axis=1, inplace=True)

# Data visualization
# sns.countplot(x='Survived', data=train_data)
# plt.show()

# sns.countplot(x='Pclass', hue='Survived', data=train_data)
# plt.show()

# Convert categorical data to numeric
train_data['Sex'] = train_data['Sex'].map({'male': 0, 'female': 1})
train_data['Embarked'] = train_data['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

# Feature selection
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X = train_data[features]
y = train_data['Survived']

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_val)

# Evaluate the model
print("Accuracy:", accuracy_score(y_val, y_pred))
# print("Confusion Matrix:\n", confusion_matrix(y_val, y_pred))
# print("Classification Report:\n", classification_report(y_val, y_pred))

# Predict on test data
test_data['Sex'] = test_data['Sex'].map({'male': 0, 'female': 1})
test_data['Embarked'] = test_data['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
test_data['Age'].fillna(test_data['Age'].median(), inplace=True)
test_data['Fare'].fillna(test_data['Fare'].median(), inplace=True)
test_data['is_alone'] = np.where((test_data['SibSp'] + test_data['Parch']) > 0, 0, 1)
test_data['FamilySize'] = test_data['SibSp'] + test_data['Parch'] + 1

X_test = test_data[features]
test_predictions = model.predict(X_test)

# Save submission
submission = pd.DataFrame({
    'PassengerId': test_data['PassengerId'],
    'Survived': test_predictions
})
submission.to_csv('submission.csv', index=False)
