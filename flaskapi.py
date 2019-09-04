from flask import Flask, jsonify, request
import pickle
import numpy as np


linear_model = 'models/linear_model.pkl'
lasso_model = 'models/lasso_model.pkl'

model = None

app = Flask(__name__)

ERROR_MESSAGE = {
    'error': 'format invalid'
}
def load_model(path):
    global model
    model = pickle.load(open(path, 'rb'))


@app.route("/")
def index():
    return "Hello World"

@app.route("/smartpricing", methods = [ 'POST'])
def predict():
    coef = model.coef_
    coef = list(coef)
    if not request.json:
        return jsonify(ERROR_MESSAGE)
    else:
        bedrooms = request.json["bedrooms"]
        bathrooms = request.json["bathrooms"]
        beds = request.json["beds"]
        maxguest = request.json["maxguest"]
        bookingType = request.json["bookingType"]
        property = request.json["property"]
        data_input = [bedrooms, bathrooms, beds, maxguest, bookingType, property]
        pred_price = model.predict(np.array([data_input]))
        result = {
            "price": float(pred_price),
            "coef": coef
        }
        return jsonify(result)



if __name__ == "__main__":
    load_model(lasso_model)
    app.run(debug=True)
