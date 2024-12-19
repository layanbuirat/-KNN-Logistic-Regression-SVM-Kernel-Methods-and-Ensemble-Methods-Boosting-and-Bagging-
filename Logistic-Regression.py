import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def load_and_clean_data(file_path):
    # Define the new column names
    column_names = [
        "Id", "SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm", "Species"
    ]

    # Load the dataset
    data = pd.read_csv(file_path, header=None, names=column_names)

    # Convert columns to numeric
    data.iloc[:, 1:-1] = data.iloc[:, 1:-1].apply(pd.to_numeric, errors='coerce')

    # Fill NaN values with the column mean
    data.iloc[:, 1:-1] = data.iloc[:, 1:-1].fillna(data.iloc[:, 1:-1].mean())

    # Convert Species to numeric
    data['Species'] = data['Species'].map({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})

    # Drop rows with missing values
    return data.dropna()

def train_logistic_regression(data):
    # Separate features and labels
    X = data.iloc[:, 1:-1].values  # Features
    y = data['Species'].values    # Labels

    # Split the dataset , 80% & 20% to test 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train Logistic Regression with L2 regularization
    model_l2 = LogisticRegression(penalty='l2', solver='liblinear', random_state=42)
    model_l2.fit(X_train, y_train)

    # Predict and evaluate for L2 regularization
    y_pred_l2 = model_l2.predict(X_test)
    print("\nLogistic Regression (L2 Regularization):")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_l2))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_l2))
    print(f"Accuracy: {accuracy_score(y_test, y_pred_l2)}")

    # Train Logistic Regression with L1 regularization
    model_l1 = LogisticRegression(penalty='l1', solver='liblinear', random_state=42)
    model_l1.fit(X_train, y_train)

    # Predict and evaluate for L1 regularization
    y_pred_l1 = model_l1.predict(X_test)
    print("\nLogistic Regression (L1 Regularization):")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_l1))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_l1))
    print(f"Accuracy: {accuracy_score(y_test, y_pred_l1)}")

if __name__ == "__main__":
    file_path = r"C:\Users\hp\Documents\projects\Machine Learning\Iris.csv"  # Update with your file path
    data = load_and_clean_data(file_path)
    train_logistic_regression(data)
