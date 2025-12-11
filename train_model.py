import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

data = pd.read_csv("Crop_recommendation.csv") 

if not os.path.exists("model.pkl"):

    X = data[["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]]  
    y = data["label"]  

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    crop_model = RandomForestClassifier(random_state=42)
    crop_model.fit(X_train, y_train)

    joblib.dump(crop_model, "model.pkl")

    print("Model trained and saved as model.pkl")

else:
    print("You already have model.pkl file so,\nRun command : python ./main.py ")