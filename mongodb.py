from pymongo import MongoClient

client = MongoClient('localhost', 27017)
db = client.smart_pricing
#collection = db.test_collection
