import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the dataset
file_path = 'cases_clean.csv'
df = pd.read_csv(file_path)

print("First few rows of the dataset:")
print(df.head())

# Drop unnecessary columns
columns_to_drop = ['Bankruptcy Proceedings Initiated', 'Bankruptcy Declared', 'Plaintiff Represented',
                   'Defendant Represented', 'Case ID', 'Date of Decision', 'Awarded Principal Amount', 
                   'Awarded Penalty Amount', 'Awarded Fine', 'Awarded Unjustly Withheld Funds', 
                   'Category of a Claim']
df = df.drop(columns=columns_to_drop)

# Convert binary columns to 1/0
binary_cols = ['Supporting Documents Presented', 'Presence of Reconciliation Act', 'Presence of Contract',
               'Presence of Invoice', 'Penalty Conditions', 'Partial Payment Made', 'Deadline Specified',
               'Presence of Demand Letter']
for col in binary_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0})

# Prepare features and target
X = df.drop(columns=['Outcome'])
y = df['Outcome']

# Encode the target variable
le = LabelEncoder()
y = le.fit_transform(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("Training feature names:", X_train.columns.tolist())  # Print feature names for verification
print(X_train.dtypes)

# Define parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

# Perform grid search
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), 
                           param_grid, 
                           cv=5, 
                           n_jobs=-1, 
                           scoring='accuracy')
grid_search.fit(X_train, y_train)

# Print results
print("Best parameters:", grid_search.best_params_)
print("Best cross-validation score:", grid_search.best_score_)
best_rf = grid_search.best_estimator_

# Evaluate the model
y_pred = best_rf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Feature importance
importances = best_rf.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
print(feature_importance_df.sort_values(by='Importance', ascending=False))

# Save the model, label encoder, and feature names
joblib.dump(best_rf, 'best_rf_model.pkl')
joblib.dump(le, 'label_encoder.pkl')
joblib.dump(X_train.columns.tolist(), 'feature_names.pkl')  # Save feature names
print("Model, encoder, and feature names saved successfully.")


# example_input = {
#     'Bankruptcy Proceedings Initiated': 1,
#     'Bankruptcy Declared': 1,
#     'Asked Principal Amount': 100000000.0,
#     'Asked Penalty Amount': 50000000.0,
#     'Asked Fine': 0.0,
#     'Asked Unjustly Withheld Funds': 0.0,
#     'Plaintiff Represented': 1,
#     'Defendant Represented': 1,
#     'Presence of Reconciliation Act': 0,
#     'Presence of Contract': 1,
#     'Presence of Invoice': 1,
#     'Penalty Conditions': 1,
#     'Total Value of Delivered Goods': 100000000.0,
#     'Partial Payment Made': 0,
#     'Deadline Specified': 1,
#     'Presence of Demand Letter': 1
# }

# input_df = pd.DataFrame([example_input], columns=feature_names)

# probabilities = best_rf.predict_proba(input_df)[0]
# outcome_probabilities = dict(zip(le.classes_, probabilities))

# prediction = best_rf.predict(input_df)
# importance_example = best_rf.feature_importances_
# outcome = le.inverse_transform(prediction)[0]

# print(importance_example)
# importances = best_rf.feature_importances_
# print(importances)
# print(f"Predicted Outcome: {outcome}")
# print("\nProbability for each possible outcome:")
# for outcome, prob in outcome_probabilities.items():
#     print(f"{outcome}: {prob:.2%}")

joblib.dump(best_rf, 'best_rf_model.pkl')
joblib.dump(le, 'label_encoder.pkl')