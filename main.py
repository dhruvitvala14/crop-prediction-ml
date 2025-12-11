import numpy as np
import joblib
from flask import Flask, request, render_template

app = Flask(__name__)

model = joblib.load("model.pkl")
print("Loaded model type:", type(model)) 

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    nitrogen = float(request.form['nitrogen'])
    phosphorus = float(request.form['phosphorus'])
    potassium = float(request.form['potassium'])
    temperature = float(request.form['temperature'])
    humidity = float(request.form['humidity'])
    ph = float(request.form['ph'])
    rainfall = float(request.form['rainfall'])

    features = np.array([[nitrogen, phosphorus, potassium,
                          temperature, humidity, ph, rainfall]])

    prediction = model.predict(features)[0]

    return render_template(
        "index.html",
        prediction_text=f"The Predicted Crop is {prediction}"
    )

if __name__ == "__main__":
    app.run(debug=True)
