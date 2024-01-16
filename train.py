# train.py
import argparse
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error

# Define the column transformer to select columns '6' and '7'
column_transformer = ColumnTransformer(
    transformers=[
        ('selected_columns', 'passthrough', ['6', '7'])
    ],
    remainder='drop'  # Drop all other columns not specified
)


def save_model(model, save_path='trained_model.pkl'):
    """
    Save the trained model to a pickle file.
    """
    with open(save_path, 'wb') as file:
        pickle.dump(model, file)
        
def load_data(file_path):
    """
    Load dataset from a given file path.
    """
    return pd.read_csv(file_path)

def train_model(X, y, degree=2):
    """
    Train a polynomial regression model.
    """
    pipeline = Pipeline([
        ("column_transformer", column_transformer),  # Select columns '6' and '7'
        ("polynomial_features", PolynomialFeatures(degree=degree)),
        ("ridge_regression", LinearRegression())
    ])
    pipeline.fit(X, y)
    return pipeline

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model using Mean Squared Error.
    """
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Train a polynomial ridge regression model.")
    parser.add_argument('file_path', type=str, help='Path to the CSV dataset file')
    
    # Parse arguments
    args = parser.parse_args()
    file_path = args.file_path
    
    # Load the dataset
    df = load_data(file_path)

    # Splitting the dataset into features and target
    X = df
    y = df.loc[:, 'target']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = train_model(X_train, y_train, degree=2)

    # Evaluate the model
    evaluate_model(model, X_test, y_test)
    
    # Train the model on all data
    model = train_model(X, y, degree=2)
    save_model(model)

if __name__ == "__main__":
    main()
