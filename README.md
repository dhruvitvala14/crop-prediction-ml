# Crop Prediction using Machine Learning & Flask 🌱

This project is a **Machine Learning–based Crop Recommendation System** built with  
**Python, scikit-learn, and Flask**.  

Given soil and climate parameters such as **Nitrogen, Phosphorus, Potassium, Temperature, Humidity, pH, and Rainfall**,  
the model predicts the **most suitable crop** to grow.

---

## 🚀 Features

- End-to-end **ML pipeline** for crop prediction
- **`train_model.py`** script to train and generate `model.pkl` (only once)
- **Flask web app (`main.py`)** with a user-friendly form
- Input validation for all features
- Clean UI built with **HTML + CSS (templates + static)**
- Ready to deploy or extend

---

## 🧠 Model Overview

- **Problem Type:** Multi-class classification  
- **Input Features:**  
  - Nitrogen  
  - Phosphorus  
  - Potassium  
  - Temperature  
  - Humidity  
  - pH  
  - Rainfall  

- **Output:** Recommended crop (e.g. rice, wheat, maize, etc.)
- **Libraries:** `pandas`, `numpy`, `scikit-learn`, `joblib`

---

## 📁 Project Structure

```text
crop-prediction-ml/
│
├── main.py              # Flask application
├── train_model.py       # Trains the model and saves model.pkl
├── model.pkl            # Trained ML model (generated once)
├── requirements.txt     # Project dependencies
├── README.md            # Project documentation
│
├── templates/
│   └── index.html       # Front-end form for user inputs
│
├── static/
│   └── style.css        # Styling for the web app
│
└── screenshots/
    └── home.png         # Screenshot of the web interface
