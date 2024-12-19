import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

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

def train_adaboost_model(data):
    # Separate features and labels
    X = data.iloc[:, 1:-1].values  # Features (columns 2 to 5, excluding Id and Species)
    y = data['Species'].values  # Labels (Species)

    # Split the dataset into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Create an AdaBoost model using DecisionTreeClassifier as the base estimator
    adaboost = AdaBoostClassifier(n_estimators=50, random_state=42)

    # Train the model
    adaboost.fit(X_train, y_train)

    # Make predictions on the testing data
    y_pred = adaboost.predict(X_test)

    # Evaluate the model
    print("\nResults using AdaBoost:")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

def main():
    # Specify the file path
    file_path = r"C:\Users\Admin\Desktop\Iris.csv"  # Update with your file path

    # Load and clean the data
    data_cleaned = load_and_clean_data(file_path)

    # Train the AdaBoost model
    train_adaboost_model(data_cleaned)

if __name__ == "__main__":
    main()


