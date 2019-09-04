import os
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split

import pymongo

# DATABASE_NAME = "smart_pricing"
# COLECTION_NAME = 'hanoi'
# MONGO_URL = '127.0.0.1'
# PORT = 27017
#
# connection = pymongo.MongoClient(MONGO_URL, PORT)
# DB = connection[DATABASE_NAME]
# TABLE = connection[COLECTION_NAME]

from pymongo import MongoClient

client = MongoClient('localhost', 27017)
db = client.smart_pricing
posts = db.hanoi

data__num_bedrooms = []
data__num_bathrooms = []
data__num_beds = []
data__maximum_guests = []
data__booking_type = []
data__property_type__data__name = []
data__price__data__nightly_price_vnd = []
data1 = []
parameters = []

def getDataMongo():
    for post in posts.find():
        a = post['data__num_bedrooms']
        data__num_bedrooms.append(a)
        b = post['data__num_bathrooms']
        data__num_bathrooms.append(b)

        c = post['data__num_beds']
        data__num_beds.append(c)

        d = post['data__maximum_guests']
        data__maximum_guests.append(d)

        e = post['data__booking_type']
        data__booking_type.append(e)

        f = post['data__property_type__data__name']
        data__property_type__data__name.append(f)

        g = post['data__price__data__nightly_price_vnd']
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
def getData():
    # Get home data from CSV file
    dataFile = None
    if os.path.exists('data-luxstay-hanoi.csv'):
        print("-- data-luxstay-hanoi.csv found locally")
        dataFile = pd.read_csv('data-luxstay-hanoi.csv', skipfooter=1, encoding="utf-8")

    return dataFile

def linearRegressionModel(X_train, Y_train, X_test, Y_test):

    linear = linear_model.LinearRegression()
    # Training process
    linear.fit(X_train, Y_train)
    # Evaluating the model
    score_trained = linear.score(X_test, Y_test)

    return score_trained

def lassoRegressionModel(X_train, Y_train, X_test, Y_test):

    # temp = [2, 2, 3, 5, 1, 5]
    lasso_linear = linear_model.Lasso(alpha=1.0)
    # Training process
    lasso_linear.fit(X_train, Y_train)

    # Evaluating the model

    score_trained = lasso_linear.score(X_test, Y_test)
    #result = lasso_linear.predict(np.array([temp]))
    print(lasso_linear.coef_)
    parameters = lasso_linear.coef_

    # return score_trained #, result
    return score_trained, parameters

def run():
    data = getDataMongo()
    if data is not None:
        # Selection few attributes
        attributes = list(
            [
                'data__num_bedrooms',
                'data__num_bathrooms',
                'data__num_beds',
                'data__maximum_guests',
                'data__booking_type',
                'data__property_type__data__name'
            ]
        )
        # Vector price of house
        Y = data['data__price__data__nightly_price_vnd']
        # print np.array(Y)
        # Vector attributes of house
        X = data[attributes]
        # print(np.shape(X))
        # print(np.shape(Y))
        # Split data to training test and testing test
        X_train, X_test, Y_train, Y_test = train_test_split(np.array(X), np.array(Y), test_size=0.2)
        # Linear Regression Model
        linearScore = linearRegressionModel(X_train, Y_train, X_test, Y_test)
        print('Linear Score = ', linearScore)
        # LASSO Regression Model
        # lassoScore, result = lassoRegressionModel(X_train, Y_train, X_test, Y_test)
        lassoScore, parameters = lassoRegressionModel(X_train, Y_train, X_test, Y_test)
        print('Lasso Score = ', lassoScore)
        # print('Gia test', result)
        num_bedrooms_pr = parameters[0]
        num_bathrooms_pr = parameters[1]
        num_beds_pr = parameters[2]
        maximum_guests_pr = parameters[3]
        booking_type_pr = parameters[4]
        property_type__data__name_pr = parameters[5]

        print(num_bedrooms_pr)
        print(num_bathrooms_pr)
        print(num_beds_pr)
        print(maximum_guests_pr)
        print(booking_type_pr)
        print(property_type__data__name_pr)

        num_bedrooms = 1
        num_bathrooms = 1
        num_beds = 1
        maximum_guests = 1
        booking_type = 0
        property_type__data__name = 1

        data__price__data__nightly_price_vnd = num_bedrooms * num_bedrooms_pr + num_bathrooms * num_bathrooms_pr + num_beds * num_beds_pr + maximum_guests * maximum_guests_pr + booking_type * booking_type_pr + property_type__data__name * property_type__data__name_pr
        print(data__price__data__nightly_price_vnd)
        result = [num_bedrooms_pr, num_bathrooms_pr, num_beds_pr, maximum_guests_pr, booking_type_pr, property_type__data__name_pr]
        return result
if __name__ == "__main__":
    run()