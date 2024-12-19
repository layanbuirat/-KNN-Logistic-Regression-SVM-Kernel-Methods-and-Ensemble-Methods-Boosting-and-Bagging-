import pandas as pd
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def load_and_clean_data(file_path):
    # Define the new column names
    column_names = [
        "Id", "SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm", "Species"
    ]

    # Load the dataset
    data = pd.read_csv(file_path, header=None, names=column_names)

    # Inspect the first few rows
    print("Original Data:")
    print(data.head())

    # Convert columns to numeric (coerce errors to NaN)
    data.iloc[:, 1:-1] = data.iloc[:, 1:-1].apply(pd.to_numeric, errors='coerce')

    # Check for NaN values
    print("\nData After Conversion to Numeric:")
    print(data.isna().sum())

    # Fill NaN values with the column mean (if necessary)
    data.iloc[:, 1:-1] = data.iloc[:, 1:-1].fillna(data.iloc[:, 1:-1].mean())

    # Check for NaN values after filling
    print("\nData After Filling NaN with Mean:")
    print(data.isna().sum())

    # Convert Species to numeric: Iris-setosa = 0, Iris-versicolor = 1, Iris-virginica = 2
    data['Species'] = data['Species'].map({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})

    # Drop rows with missing values
    print(f"\nOriginal dataset had {data.shape[0]} rows.")
    data_cleaned = data.dropna()
    print(f"After cleaning, {data_cleaned.shape[0]} rows remain.")

    return data_cleaned

def determine_optimal_k(data):
    # Separate features and labels
    X = data.iloc[:, 1:-1].values  # Features (columns 2 to 5, excluding Id and Species)
    y = data['Species'].values  # Labels (Species)

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Range of K values to try
    k_values = range(1, 21)  # Trying values from 1 to 20 for K

    # List to store cross-validation scores for each K
    cv_scores = []

    # Perform cross-validation for each K
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X_scaled, y, cv=5, scoring='accuracy')  # 5-fold cross-validation
        cv_scores.append(scores.mean())  # Store average score for each K

    # Find the K with the highest cross-validation score
    optimal_k = k_values[np.argmax(cv_scores)]
    print(f"\nOptimal K (Number of Neighbors) based on cross-validation: {optimal_k}")
    
    # Plotting the cross-validation scores for each K (optional)
    import matplotlib.pyplot as plt
    plt.plot(k_values, cv_scores, marker='o')
    plt.xlabel('Number of Neighbors (K)')
    plt.ylabel('Cross-Validation Accuracy')
    plt.title('Cross-Validation Accuracy for Different K values')
    plt.show()

    return optimal_k

def train_knn_model(data, distance_metric):
    # Separate features and labels
    X = data.iloc[:, 1:-1].values  # Features (columns 2 to 5, excluding Id and Species)
    y = data['Species'].values  # Labels (Species)

    # Split the dataset into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Create and fit the KNN model with specified distance metric
    knn = KNeighborsClassifier(n_neighbors=5, metric=distance_metric)
    knn.fit(X_train, y_train)

    # Make predictions on the testing data
    y_pred = knn.predict(X_test)

    # Evaluate the model
    print(f"\nResults using {distance_metric} distance metric:")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

def main():
    # Specify the file path
    file_path = r"C:\Users\Admin\Desktop\Iris.csv"  # Update with your file path

    # Load and clean the data
    data_cleaned = load_and_clean_data(file_path)
    

    # Experiment with different distance metrics
    for metric in ['euclidean', 'manhattan', 'cosine']:
        train_knn_model(data_cleaned, metric)

    optimal_k = determine_optimal_k(data_cleaned)
    


if __name__ == "__main__":
    main()
