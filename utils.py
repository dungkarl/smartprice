"""
    file chứa function xử lý data và build model
"""
import os
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import pymongo
import pickle
from connect import load_json_data




data_path = 'luxstayjson.json'
data_list = load_json_data(data_path)
data__num_bedrooms = []
data__num_bathrooms = []
data__num_beds = []
data__maximum_guests = []
data__booking_type = []
data__property_type__data__name = []
data__price__data__nightly_price_vnd = []
#data1 = []
parameters = []

def convert_data():
    for obj_data in data_list:
        a = obj_data['data__num_bedrooms']
        data__num_bedrooms.append(a)
        b = obj_data['data__num_bathrooms']
        data__num_bathrooms.append(b)

        c = obj_data['data__num_beds']
        data__num_beds.append(c)

        d = obj_data['data__maximum_guests']
        data__maximum_guests.append(d)

        e = obj_data['data__booking_type']
        data__booking_type.append(e)

        f = obj_data['data__property_type__data__name']
        data__property_type__data__name.append(f)

        g = obj_data['data__price__data__nightly_price_vnd']
        data__price__data__nightly_price_vnd.append(g)

    my_data = {
        "data__num_bedrooms": data__num_bedrooms,
        "data__num_bathrooms": data__num_bathrooms,
        "data__num_beds": data__num_beds,
        "data__maximum_guests": data__maximum_guests,
        "data__booking_type": data__booking_type,
        "data__property_type__data__name": data__property_type__data__name,
        "data__price__data__nightly_price_vnd": data__price__data__nightly_price_vnd
    }
    my_data = pd.DataFrame(my_data)
    #data1 = [data__num_bedrooms, data__num_bathrooms, data__num_beds, data__maximum_guests, data__booking_type, data__property_type__data__name]
    return my_data


def linearRegressionModel(X_train, y_train, X_test, y_test):
    """
        include:
        LinearRegression Model
        model name
        return model
    """
    model_name = 'models/linear_model.pkl'
    ln_model = linear_model.LinearRegression()
    ln_model.fit(X_train, y_train)
    score_trained = ln_model.score(X_test, y_test)
    print("score of linear model:", score_trained)
    pickle.dump(ln_model, open(model_name, 'wb'))
    return "Trained linear model"

def lassoRegressionModel(X_train, y_train, X_test, y_test):
    """

    """
    model_name = 'models/lasso_model.pkl'
    lasso_model = linear_model.Lasso(alpha=1.0)
    lasso_model.fit(X_train, y_train)
    score_trained = lasso_model.score(X_test, y_test)
    print("score of lasso model:", score_trained)
    pickle.dump(lasso_model, open(model_name, 'wb'))
    return "Trained Lasso model"

def run():
    data = convert_data()
    attributes = [
        'data__num_bedrooms',
        'data__num_bathrooms',
        'data__num_beds',
        'data__maximum_guests',
        'data__booking_type',
        'data__property_type__data__name'
    ]
    if data is not None:
        X = data[attributes]
        y = data['data__price__data__nightly_price_vnd']
    X_train, X_test, y_train, y_test = train_test_split(np.array(X), np.array(y), test_size=0.2)
    linearRegressionModel(X_train, y_train, X_test, y_test)
    lassoRegressionModel(X_train, y_train, X_test, y_test)
    #print(data.head())
    print(X.head())

if __name__ == '__main__':
    run()
