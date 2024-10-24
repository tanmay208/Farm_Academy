import pickle
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# Load the pickled RandomForest model
with open('RandomForest.pkl', 'rb') as file:
    model = pickle.load(file)

# Example: Data (X) and Labels (y) - replace with your actual data
# Feature data with 7 inputs (7 features)
X = np.array([
    [5.1, 3.5, 1.4, 0.2, 0.5, 0.3, 1.2],  # Example feature 1
    [4.9, 3.0, 1.4, 0.2, 0.6, 0.4, 1.3],  # Example feature 2
    [6.2, 3.4, 5.4, 2.3, 0.7, 0.5, 1.5],  # Example feature 3
    [5.9, 3.0, 5.1, 1.8, 0.8, 0.6, 1.6],  # Example feature 4
    [5.0, 3.6, 1.4, 0.2, 0.5, 0.3, 1.2],  # Example feature 5
    [6.7, 3.1, 4.7, 1.5, 0.9, 0.7, 1.8],  # Example feature 6
    [5.5, 2.3, 4.0, 1.3, 0.4, 0.4, 1.1],  # Example feature 7
])  

# True labels (ensure these match your model's output)
y = np.array(['kidneybeans', 'kidneybeans', 'blackbeans', 'blackbeans', 'kidneybeans', 'blackbeans', 'kidneybeans'])  # Example labels

# Split the dataset into training and test sets (replace this with your actual data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Check the predictions
print("Predictions:", y_pred)
print("True Labels:", y_test)

# Calculate and print accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Print classification report for detailed metrics
print(classification_report(y_test, y_pred))
