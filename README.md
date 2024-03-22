# Rain Prediction using Random Forest

This project uses Random Forest Classifier to predict whether it will rain tomorrow based on various weather factors. The data used in this project is obtained from a CSV file located at `../resources/weatherAUS_clean.csv`.

## Requirements

- Python (version 3.8 or higher)
- joblib
- pandas
- matplotlib
- numpy
- scikit-learn

## Installation

1. Make sure you have Python installed. You can download it from [python.org](https://www.python.org/downloads/).

2. Install the required dependencies using pip:
3. Clone or download this repository to your local machine.

4. Navigate to the project directory.

5. Run the Python script `rain_prediction.py`.

## Usage

The script `rain_prediction.py` performs the following steps:

1. Loads the dataset from the CSV file.
2. Splits the dataset into features and labels.
3. Converts categorical variables into dummy variables.
4. Splits the dataset into training and testing sets.
5. Normalizes the numerical columns using StandardScaler.
6. Trains a Random Forest Classifier on the training data.
7. Makes predictions on the testing data.
8. Evaluates the performance of the model.
9. Saves the trained model to a file.
10. Prints the results and displays a bar chart showing the number of correct predictions for "Yes" and "No" rain tomorrow.

## Contribution

If you'd like to contribute to this project, feel free to fork this repository, make your changes, and submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

For any questions or feedback regarding this project, please contact javierlink22@gmail.com.

