import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Preprocess the data
# Handle missing values, encode categorical variables, and scale features

# Split the data into features and target
X = train_data.drop(columns=['Transported'])
y = train_data['Transported']

# Extract PassengerId from test data
test_passenger_ids = test_data['PassengerId']

# One-hot encode categorical variables
X = pd.get_dummies(X)
test_data = pd.get_dummies(test_data.drop(columns=['PassengerId']))

# Align the test data columns with the training data columns
test_data = test_data.reindex(columns=X.columns, fill_value=0)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_val)

# Evaluate the model
accuracy = accuracy_score(y_val, y_pred)
print(f'Validation Accuracy: {accuracy:.2f}')

# Predict on test data
predictions = model.predict(test_data)

# Create a DataFrame for submission
submission = pd.DataFrame({
    'PassengerId': test_passenger_ids,
    'Transported': predictions
})

# Convert boolean predictions to the required format (True/False)
submission['Transported'] = submission['Transported'].astype(bool)

# Save the DataFrame to a CSV file
submission.to_csv('submission.csv', index=False)