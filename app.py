from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("sales_model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    marketing = float(request.form["marketing"])
    customers = float(request.form["customers"])
  

    features = np.array([[marketing, customers
                          ]])

    prediction = model.predict(features)

    return render_template(
        "index.html",
        prediction_text=f"Predicted Sales: {prediction[0]:.2f}"
    )

if __name__ == "__main__":
    app.run(debug=True)