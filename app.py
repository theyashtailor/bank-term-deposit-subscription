from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load the trained model
model = joblib.load("gradient_boosting_model.pkl")

# Initialize Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return "Gradient Boosting Model API is running! 🚀"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = np.array(data["features"]).reshape(1, -1)  # Ensure input is a 2D array

        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0].tolist()

        return jsonify({"prediction": int(prediction), "probability": probability})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
