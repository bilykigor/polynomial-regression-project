# predict.py
import pandas as pd
import pickle
import argparse

def load_model(model_path):
    """
    Load the trained model from a pickle file.
    """
    with open(model_path, 'rb') as file:
        loaded_model = pickle.load(file)
    return loaded_model

def load_test_data(test_data_path):
    """
    Load test data from a CSV file.
    """
    test_data = pd.read_csv(test_data_path)
    return test_data

def make_predictions(model, test_data):
    """
    Use the model to make predictions on the test data.
    """
    predictions = model.predict(test_data)
    return predictions

def save_predictions(predictions):
    """
    Save the predictions to a CSV file.
    """
    pd.DataFrame({"Prediction": predictions}).to_csv('predictions.csv', index=False)

def main():
    parser = argparse.ArgumentParser(description='Make predictions with a trained model.')
    parser.add_argument('model_path', type=str, help='Path to the trained model file')
    parser.add_argument('test_data_path', type=str, help='Path to the test data CSV file')

    args = parser.parse_args()

    # Load the model and test data
    model = load_model(args.model_path)
    test_data = load_test_data(args.test_data_path)

    # Make predictions
    predictions = make_predictions(model, test_data)
    
    # Save predictions
    save_predictions(predictions)

if __name__ == "__main__":
    main()
