# Titanic Survival Prediction
# Author: Abhishek Shah
# Date: 2025-02-17

# Importing the required libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import warnings
import re

warnings.filterwarnings('ignore')

# Data Preprocessing Section

# Load the dataset
def load_data(file_path):
    """
    Load the Titanic dataset from a CSV file.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded data with {df.shape[0]} rows and {df.shape[1]} columns")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# Data preprocessing
def preprocess_data(df):
    """
    Preprocess the Titanic dataset by handling missing values, 
    feature engineering, and encoding categorical variables.
    """
    # Create a copy of the dataframe
    data = df.copy()
    
    # Extract title from Name before dropping the column
    # This regex extracts the title (e.g., Mr, Mrs, Miss, etc.)
    data['Title'] = data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    
    # Drop unnecessary columns
    data = data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    
    # Impute missing values for Embarked if any
    if data['Embarked'].isnull().sum() > 0:
        data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])
    
    # Create FamilySize feature
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
    data['IsAlone'] = (data['FamilySize'] == 1).astype(int)
    
    # Handle missing Age values and create AgeGroup feature
    data['Age'] = data['Age'].fillna(data['Age'].median())
    data['AgeGroup'] = pd.cut(data['Age'], bins=[0, 12, 18, 65, 100], 
                              labels=['Child', 'Teen', 'Adult', 'Senior'])
    
    # Handle missing Fare values and create FareGroup feature
    data['Fare'] = data['Fare'].fillna(data['Fare'].median())
    try:
        data['FareGroup'] = pd.qcut(data['Fare'], q=4, labels=['Low', 'Mid', 'High', 'Very High'])
    except ValueError:
        # In case there are duplicate edges, use cut instead
        data['FareGroup'] = pd.cut(data['Fare'], bins=4, labels=['Low', 'Mid', 'High', 'Very High'])
    
    # Define features and target
    X = data.drop('Survived', axis=1)
    y = data['Survived']
    
    # Identify categorical and numerical features
    categorical_features = ['Sex', 'Embarked', 'Title', 'AgeGroup', 'FareGroup']
    numerical_features = ['Age', 'Fare', 'FamilySize', 'IsAlone', 'SibSp', 'Parch']
    
    # Create preprocessing pipelines for numerical and categorical features
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return X, y, preprocessor

# Model Training and Evaluation Section

# Train and evaluate models
def train_and_evaluate_models(X, y, preprocessor):
    """
    Train and evaluate multiple machine learning models.
    """
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Initialize results dictionary
    results = {}
    
    # 1. Logistic Regression
    print("Training Logistic Regression model...")
    lr_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])
    
    lr_param_grid = {
        'classifier__C': [0.01, 0.1, 1, 10],
        'classifier__penalty': ['l2']
    }
    
    lr_grid = GridSearchCV(lr_pipeline, lr_param_grid, cv=5, scoring='accuracy')
    lr_grid.fit(X_train, y_train)
    
    y_pred_lr = lr_grid.predict(X_test)
    results['Logistic Regression'] = {
        'model': lr_grid,
        'accuracy': accuracy_score(y_test, y_pred_lr),
        'precision': precision_score(y_test, y_pred_lr),
        'recall': recall_score(y_test, y_pred_lr),
        'f1': f1_score(y_test, y_pred_lr),
        'confusion_matrix': confusion_matrix(y_test, y_pred_lr),
        'y_pred': y_pred_lr
    }
    
    # 2. Random Forest
    print("Training Random Forest model...")
    rf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    rf_param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [None, 5, 10],
        'classifier__min_samples_split': [2, 5]
    }
    
    rf_grid = GridSearchCV(rf_pipeline, rf_param_grid, cv=5, scoring='accuracy')
    rf_grid.fit(X_train, y_train)
    
    y_pred_rf = rf_grid.predict(X_test)
    results['Random Forest'] = {
        'model': rf_grid,
        'accuracy': accuracy_score(y_test, y_pred_rf),
        'precision': precision_score(y_test, y_pred_rf),
        'recall': recall_score(y_test, y_pred_rf),
        'f1': f1_score(y_test, y_pred_rf),
        'confusion_matrix': confusion_matrix(y_test, y_pred_rf),
        'y_pred': y_pred_rf
    }
    
    # 3. Neural Network
    print("Training Neural Network model...")
    # Prepare data for neural network
    preprocessor.fit(X_train)
    X_train_nn = preprocessor.transform(X_train)
    X_test_nn = preprocessor.transform(X_test)
    
    # Create a neural network model
    nn_model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train_nn.shape[1],)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    
    # Compile the model
    nn_model.compile(optimizer='adam',
                     loss='binary_crossentropy',
                     metrics=['accuracy'])
    
    # Early stopping callback
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # Train the neural network model
    history = nn_model.fit(
        X_train_nn, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=0
    )
    
    # Evaluate the neural network model
    y_pred_nn_prob = nn_model.predict(X_test_nn)
    y_pred_nn = (y_pred_nn_prob > 0.5).astype(int).ravel()
    
    results['Neural Network'] = {
        'model': nn_model,
        'accuracy': accuracy_score(y_test, y_pred_nn),
        'precision': precision_score(y_test, y_pred_nn),
        'recall': recall_score(y_test, y_pred_nn),
        'f1': f1_score(y_test, y_pred_nn),
        'confusion_matrix': confusion_matrix(y_test, y_pred_nn),
        'y_pred': y_pred_nn,
        'history': history
    }
    
    return results, X_test, y_test

# Visualization and Reporting Section
# Visualize results
def visualize_results(results, X_test, y_test):
    """
    Visualize and compare model performance.
    """
    # Comparison of accuracy, precision, recall, and F1 score
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    models = list(results.keys())
    
    df_metrics = pd.DataFrame(index=models, columns=metrics)
    
    for model in models:
        for metric in metrics:
            df_metrics.loc[model, metric] = results[model][metric]
    
    # Plot metrics comparison
    plt.figure(figsize=(12, 6))
    df_metrics.plot(kind='bar', figsize=(12, 6))
    plt.title('Model Performance Comparison')
    plt.ylabel('Score')
    plt.xlabel('Model')
    plt.xticks(rotation=0)
    plt.ylim(0, 1)
    plt.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.show()
    
    # Plot confusion matrices for each model
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5))
    
    if len(models) == 1:
        axes = [axes]  # Ensure axes is iterable if only one subplot exists
    
    for i, model in enumerate(models):
        cm = results[model]['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[i])
        axes[i].set_title(f'{model} Confusion Matrix')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig('confusion_matrices.png')
    plt.show()
    
    # Plot learning curve for Neural Network if available
    if 'history' in results['Neural Network']:
        history = results['Neural Network']['history']
        
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Neural Network Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='lower right')
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Neural Network Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
        
        plt.tight_layout()
        plt.savefig('nn_learning_curve.png')
        plt.show()


# Generate performance summary
def generate_summary(results, y_test):
    """
    Generate summary of model performance.
    """
    models = list(results.keys())
    summary = {}
    
    # Find the best model for each metric
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    best_models = {}
    
    for metric in metrics:
        metric_values = {model: results[model][metric] for model in models}
        best_model = max(metric_values, key=metric_values.get)
        best_value = metric_values[best_model]
        best_models[metric] = {'model': best_model, 'value': best_value}
    
    # Detailed report for each model
    for model in models:
        summary[model] = {
            'accuracy': results[model]['accuracy'],
            'precision': results[model]['precision'],
            'recall': results[model]['recall'],
            'f1': results[model]['f1'],
            'classification_report': classification_report(y_test, results[model]['y_pred'])
        }
    
    summary['best_models'] = best_models
    return summary


# Main function
def main():
    """
    Main function to execute the pipeline for Titanic survival prediction.
    """
    # Load data
    df = load_data('titanic.csv')
    if df is None:
        return
    
    # Exploratory Data Analysis
    print("Data overview:")
    print(df.info())
    print("\nMissing values:")
    print(df.isnull().sum())
    
    # Preprocess data
    X, y, preprocessor = preprocess_data(df)
    
    # Train and evaluate models
    results, X_test, y_test = train_and_evaluate_models(X, y, preprocessor)
    
    # Visualize results
    visualize_results(results, X_test, y_test)
    
    # Generate summary
    summary = generate_summary(results, y_test)
    
    # Print summary
    print("\n====== MODEL PERFORMANCE SUMMARY ======")
    for model in results.keys():
        print(f"\n{model}:")
        print(f"Accuracy: {summary[model]['accuracy']:.4f}")
        print(f"Precision: {summary[model]['precision']:.4f}")
        print(f"Recall: {summary[model]['recall']:.4f}")
        print(f"F1 Score: {summary[model]['f1']:.4f}")
        print("\nClassification Report:")
        print(summary[model]['classification_report'])
    
    print("\n====== BEST MODELS ======")
    for metric, details in summary['best_models'].items():
        print(f"Best model for {metric}: {details['model']} with {metric} = {details['value']:.4f}")

if __name__ == "__main__":
    main()
