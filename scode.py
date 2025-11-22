import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

# Sample data loading function
def load_data(filepath):
    # Load your data here (for example, a CSV file)
    return pd.read_csv(filepath)

# Data preprocessing function
def preprocess_data(df):
    # Handle missing values
    df.fillna(df.mean(), inplace=True)  # Simple mean imputation for numerical features
    # Encode categorical variables and scale numerical ones
    categorical_features = ['gender', 'diagnosis_code']  # Sample categorical features
    numerical_features = ['age', 'length_of_stay']  # Sample numerical features

    # Define preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(), categorical_features)
        ])

    X = df.drop('readmission_within_30_days', axis=1)
    y = df['readmission_within_30_days']
    return X, y, preprocessor
def train_and_evaluate_model(X, y, preprocessor):
    # Define model
    model = RandomForestClassifier(random_state=42)

    # Set up the pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                ('smote', SMOTE(random_state=42)),  # Handle class imbalance
                                ('classifier', model)])

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit the model
    pipeline.fit(X_train, y_train)

    # Make predictions
    y_pred = pipeline.predict(X_test)

    # Evaluate the model
    print(f'Accuracy: {accuracy_score(y_test, y_pred):.2f}')
    print('Classification Report:')
    print(classification_report(y_test, y_pred))
    print('Confusion Matrix:')
    print(confusion_matrix(y_test, y_pred))
    print('Confusion Matrix:')
if __name__ == "__main__":
    # Load data
    data = load_data('hospital_readmission_data.csv')  # Replace with your actual data path

    # Data preprocessing
    X, y, preprocessor = preprocess_data(data)

    # Train and evaluate the model
    train_and_evaluate_model(X, y, preprocessor)
    # Train and evaluate the model
    train_and_evaluate_model(X, y)
