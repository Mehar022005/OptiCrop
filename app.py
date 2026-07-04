import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load the trained crop-recommendation model.
# Replace model.pkl with your own trained model any time — the app doesn't
# need any other changes as long as it exposes a .predict() method that
# accepts a 2D array of [nitrogen, phosphorous, potassium, temperature,
# humidity, ph, rainfall] and returns the crop name.
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/findyourcrop')
def findyourcrop():
    return render_template('findyourcrop.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        int_features = [float(x) for x in request.form.values()]
        features = [np.array(int_features)]
        prediction = model.predict(features)
        output = prediction[0]
        prediction_text = 'Best crop for the given conditions is {}'.format(output)
    except ValueError:
        prediction_text = 'Please enter valid numeric values for all fields.'

    return render_template('findyourcrop.html', prediction_text=prediction_text)


if __name__ == "__main__":
    app.run(debug=True)
