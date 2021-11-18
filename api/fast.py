from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from datetime import datetime
import pytz

import TaxiFareModel.predict as TFMpred

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@app.get("/")
def index():
    return {"greeting": "Hello world"}

@app.get("/predict")
def predict(pickup_datetime, pickup_longitude, pickup_latitude,
            dropoff_longitude, dropoff_latitude, passenger_count):
    # create a datetime object from the user provided datetime
    pickup_datetime = datetime.strptime(pickup_datetime, "%Y-%m-%d %H:%M:%S")
    # localize the user datetime with NYC timezone and convert to UTC and object
    eastern = pytz.timezone("US/Eastern")
    formatted_pickup_datetime = eastern.localize(pickup_datetime, is_dst=None) \
                                .astimezone(pytz.utc) \
                                .strftime("%Y-%m-%d %H:%M:%S UTC")

    params_dict = {
        "key": "2013-07-06 17:18:00.000000119", # Not returned, but model requires
                                                # a key - hardcoded here
        "pickup_datetime":   [formatted_pickup_datetime],
        "pickup_longitude":  [float(pickup_longitude)],
        "pickup_latitude":   [float(pickup_latitude)],
        "dropoff_longitude": [float(dropoff_longitude)],
        "dropoff_latitude":  [float(dropoff_latitude)],
        "passenger_count":   [int(passenger_count)]
    }
    X_pred = pd.DataFrame.from_dict(params_dict)
    y_pred = TFMpred.predict(X_pred)
    return {
        "prediction": y_pred[0]
    }

if __name__ == '__main__':
    y_pred = predict(pickup_datetime="2013-07-06 17:18:00",
            pickup_longitude="-73.950655",
            pickup_latitude="40.783282",
            dropoff_longitude="-73.984365",
            dropoff_latitude="40.769802",
            passenger_count="1")
    # params_dict = {
    #     "key": "2013-07-06 17:18:00.000000119",  # Not returned, but model requires
    #                                             # a key - hardcoded here
    #     "pickup_datetime": ["2013-07-06 21:18:00 UTC"],
    #     "pickup_longitude": [float(-73.950655)],
    #     "pickup_latitude": [float(40.783282)],
    #     "dropoff_longitude": [float(-73.984365)],
    #     "dropoff_latitude": [float(40.769802)],
    #     "passenger_count": [int(1)]
    # }
    # X_pred = pd.DataFrame.from_dict(params_dict)
    print(y_pred)
