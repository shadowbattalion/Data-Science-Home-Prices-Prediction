import json
import pickle
import numpy as np

__locations = None
__data_columns = None
__model = None

def get_estimate_price(location,sqft,bath,bhk):
    load_saved_artifacts()
    try:
       loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1

    #plugging in the numbers to predict the housing price for a location
    x=np.zeros(len(__data_columns)) # creating a new np array using the length of  X number of columns and intitiating it to zeroes.
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0: # for the one hot encoding columns (locations), 
        x[loc_index] = 1 # only a given location will be marked with 1. the rest will be 0.0
    # x will look like this: sqft + bath + bhk + location

    return round(__model.predict([x])[0], 2) # will get a 2d array back. so need to call [0]

def get_location_names():
    load_saved_artifacts() ### not shown in tutorial. ask shaik about this
    return __locations

def load_saved_artifacts():
    print("loading saved artifacts...start")
    global __data_columns
    global __locations
    global __model

    with open("./artifacts/columns.json","r") as f:
        __data_columns = json.load(f)["data_columns"]
        __locations = __data_columns[3:]

    with open("./artifacts/Bangalore_home_prices_model.pickle","rb") as f: #rb because model file is binary
        __model = pickle.load(f)
        print("loading saved artifacts...done")

if __name__=="__main__":
    load_saved_artifacts()
    print(get_location_names())

    print(get_estimate_price('1st Phase JP Nagar',1000,3,2))
    print(get_estimate_price('1st Phase JP Nagar',1000,2,2))
    print(get_estimate_price('Kalhalli',1000,2,2))
    print(get_estimate_price('Ejipura',1000,2,2))


