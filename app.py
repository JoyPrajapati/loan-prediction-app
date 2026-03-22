from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("loan_status_predict")

@app.route('/')
def welcome():
    return render_template('index.html')

@app.route('/predict', methods=["POST"])
def predict():
    int_features = [float(x) for x in request.form.values()]
    features = np.array(int_features).reshape(1, -1)
    prediction = model.predict(features)

    if prediction[0] == 1:
        result = "Loan Approved"
    else:
        result = "Loan Not Approved"

    return render_template('index.html', prediction_text=result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)