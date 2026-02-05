import pandas as pd
import os
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
import joblib

# Checking for file trained pickle file existance
if not os.path.exists("model.pkl"):

    # Load dataset
    df = pd.read_csv('Crop_recommendation.csv')

    # Seprating featuers and labels
    X = df.drop(['label'], axis=1)
    y = df['label']

    # Seprating numerical and categorical columns
    numerical_features = X.select_dtypes(exclude='object').columns
    categorical_features  = X.select_dtypes(include='object').columns

    # Numerical Preprocessing
    numerical_transformer = Pipeline(steps=[
    ('scaling', StandardScaler())
    ])

    # Categorical Preprocessing
    categorical_transformer = Pipeline(steps=[
    ('ohe', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Transform all the featuers 
    preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transformer, numerical_features),
    ('cat', categorical_transformer, categorical_features)
    ], remainder='drop')

    # Model
    model = RandomForestClassifier()

    # Final Pipeline
    final_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
    ])

    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Transformed Y by label encoder
    le = LabelEncoder()
    y_train_transformed = le.fit_transform(y_train)
    y_test_transformed = le.transform(y_test)

    # Train Model
    final_model.fit(X_train, y_train_transformed)

    # Dump model into .pkl file
    joblib.dump((final_model, le), "model.pkl")

    print("Model trained and saved as model.pkl")

else:
    print("You already have model.pkl file so,\nRun command : python ./main.py ")
