# Titanic Survival Prediction
This project uses three machine learning models (Logistic Regression, Random Forest, and Neural Network) to predict the survival of Titanic passengers.

## Overview
The project follows these steps:
1. Loads and preprocesses the Titanic dataset
2. Train 3 different machine learning models
3. Evaluates model performance using accuracy, precision, recall, and F1-score
4. Compare model performance and then visualize it
5. Document findings

## Requirements
To run this project, install the dependencies from requirements.txt fiel.
You can install the dependencies using:

>> pip install -r requirements.txt


## Dataset
The project uses the classic Titanic dataset, named `titanic.csv` and placed in the project root directory. You can download it from [GitHub](https://github.com/datasciencedojo/datasets/blob/master/titanic.csv#L1).


## How to Run
1. Ensure all dependencies are installed
2. Place the `titanic.csv` file in the project root directory
3. Run the main script:

>> python index.py

## Code Structure
- `load_data()`: Loads the dataset from a CSV file
- `preprocess_data()`: Handles missing values, feature engineering, and preprocessing
- `train_and_evaluate_models()`: Trains Logistic Regression, Random Forest, and Neural Network models
- `visualize_results()`: Creates visual comparisons of model performance
- `generate_summary()`: Produces detailed metrics and identifies best models
- `main()`: Orchestrates the entire process

## Model Details
### Logistic Regression
- Uses Grid Search for hyperparameter tuning
- Optimizes for C parameter and penalty type

### Random Forest
- Uses Grid Search for hyperparameter tuning
- Optimizes for number of estimators, max depth, and min samples split

### Neural Network (TensorFlow)
- Architecture: 2 hidden layers (64 and 32 neurons) with dropout
- Uses early stopping to prevent overfitting
- Binary cross-entropy loss function

## Performance Evaluation
Models are evaluated based on:
- Accuracy: Overall correctness
- Precision: Positive predictive value
- Recall: Sensitivity/True Positive Rate
- F1-score: Harmonic mean of precision and recall
- Confusion Matrix: Visual representation of prediction errors


## Results
After running the code, the summary of results is printed to the console, and visualization plots will be saved in the outputs directory. The best performing model for each metric will also be highlighted in the end. 
