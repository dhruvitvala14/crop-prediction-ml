import pandas as pd
import joblib
from flask import Flask, request, render_template

app = Flask(__name__)

model, le = joblib.load("model.pkl")
print("Model and encoder loaded successfully")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form values
        nitrogen = float(request.form['nitrogen'])
        phosphorus = float(request.form['phosphorus'])
        potassium = float(request.form['potassium'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        columns = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

        features = pd.DataFrame(
            [[nitrogen, phosphorus, potassium,
              temperature, humidity, ph, rainfall]],
            columns=columns
        )

        prediction_encoded = model.predict(features)[0]

        prediction = le.inverse_transform([prediction_encoded])[0]

        return render_template(
            "index.html",
            prediction_text=f"The Predicted Crop is {prediction}"
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"Error: {str(e)}"
        )

if __name__ == "__main__":
    app.run(debug=True)
