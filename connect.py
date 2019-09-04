"""
    connection to json file or mongodb
"""
import json
import pymongo

# json_path = 'luxstayjson.json'
# with open(json_path, 'r') as file:
#     data_list = json.load(file)


def load_json_data(path):
    """
        return list data from json file
    """
    with open(path, 'r') as file:
        data_list = json.load(file)
    return data_list
