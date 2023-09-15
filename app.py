import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)
## load the model
model = pickle.load(open('regression_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['data']
    prediction = model.predict(scaler.transform(np.array(list(data.values())).reshape(1, -1)))  

    return jsonify(prediction[0])

## for web app
@app.route('/predict_web', methods=['POST'])
def predict_web():
    data = [float(i) for i in request.form.values()]
    final_input = scaler.transform(np.array(data).reshape(1, -1))
    print(final_input)
    model_prediction = model.predict(final_input)

    return render_template('home.html', prediction_text='Predicted Price: {}'.format(model_prediction[0]))

if __name__ == '__main__':
    app.run(debug=True)