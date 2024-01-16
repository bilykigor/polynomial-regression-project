
# Polynomial Regression Project

This project demonstrates the training of a polynomial regression model using scikit-learn and making predictions with the trained model. The project includes two main scripts: `train.py` for training the model and `predict.py` for making predictions on test data.

## Table of Contents
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [License](#license)

## Getting Started

### Prerequisites
Before running the scripts, ensure you have Python installed. You will also need to install the required dependencies listed in the `requirements.txt` file.

### Project Setup
1. Clone this repository to your local machine.
2. Navigate to the project directory:
   ```shell
   cd polynomial-regression-project
   ```

3. Install the project dependencies using pip:
   ```shell
   pip install -r requirements.txt
   ```

## Usage

### Training the Model
To train a polynomial regression model on your dataset, use the `train.py` script. Replace `your_dataset.csv` with your dataset file in CSV format (ensure the last column is the target variable). You can also adjust the polynomial degree as needed.
```shell
python train.py your_dataset.csv
```

The trained model will be saved to a pickle file named `trained_model.pkl`.

### Making Predictions
To make predictions on test data using the trained model, use the `predict.py` script. Replace `your_test_data.csv` with your test data in CSV format.
```shell
python predict.py trained_model.pkl your_test_data.csv
```

The predictions will be saved to a CSV file named `predictions.csv`.

## Project Structure
```
polynomial-regression-project/
│
├── train.py           # Script for training the model
├── predict.py         # Script for making predictions
├── trained_model.pkl  # Trained model file
├── predictions.csv    # Predictions CSV file
├── README.md          # Project README
├── requirements.txt   # Project dependencies
└── your_dataset.csv   # Your dataset file (replace with your data)
```

## Dependencies
- See `requirements.txt` for a list of required dependencies and their versions.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
