import os
from math import sqrt

import joblib
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error

from TaxiFareModel.params import *

PATH_TO_LOCAL_MODEL = 'model.joblib'

AWS_BUCKET_TEST_PATH = "s3://wagon-public-datasets/taxi-fare-test.csv"

def predict(X_pred):
    pipeline = get_model(PATH_TO_LOCAL_MODEL)
    if "best_estimator_" in dir(pipeline):
        y_pred = pipeline.best_estimator_.predict(X_pred)
    else:
        y_pred = pipeline.predict(X_pred)
    # params_dict = {
    #     "key":
    #     "2013-07-06 17:18:00.000000119",  # Not returned, but model requires
    #     # a key - hardcoded here
    #     "pickup_datetime": ["2013-07-06 21:18:00 UTC"],
    #     "pickup_longitude": [float(-73.950655)],
    #     "pickup_latitude": [float(40.783282)],
    #     "dropoff_longitude": [float(-73.984365)],
    #     "dropoff_latitude": [float(40.769802)],
    #     "passenger_count": [int(1)]
    # }
    # X_pred = pd.DataFrame.from_dict(params_dict)

    return y_pred


def get_test_data(nrows, data="s3"):
    """method to get the test data (or a portion of it) from google cloud bucket
    To predict we can either obtain predictions from train data or from test data"""
    # Add Client() here
    path = "data/test.csv"  # ⚠️ to test from actual KAGGLE test set for submission

    if data == "local":
        df = pd.read_csv(path)
    elif data == "full":
        df = pd.read_csv(AWS_BUCKET_TEST_PATH)
    else:
        df = pd.read_csv(AWS_BUCKET_TEST_PATH, nrows=nrows)
    return df


def get_model(path_to_joblib):
    pipeline = joblib.load(path_to_joblib)
    return pipeline


def evaluate_model(y, y_pred):
    MAE = round(mean_absolute_error(y, y_pred), 2)
    RMSE = round(sqrt(mean_squared_error(y, y_pred)), 2)
    res = {'MAE': MAE, 'RMSE': RMSE}
    return res


def generate_submission_csv(nrows, kaggle_upload=False):
    df_test = get_test_data(nrows)
    pipeline = joblib.load(PATH_TO_LOCAL_MODEL)
    if "best_estimator_" in dir(pipeline):
        y_pred = pipeline.best_estimator_.predict(df_test)
    else:
        y_pred = pipeline.predict(df_test)
    df_test["fare_amount"] = y_pred
    df_sample = df_test[["key", "fare_amount"]]
    name = f"predictions_test_ex.csv"
    df_sample.to_csv(name, index=False)
    print("prediction saved under kaggle format")
    # Set kaggle_upload to False unless you install kaggle cli
    if kaggle_upload:
        kaggle_message_submission = name[:-4]
        command = f'kaggle competitions submit -c new-york-city-taxi-fare-prediction -f {name} -m "{kaggle_message_submission}"'
        os.system(command)


if __name__ == '__main__':
    nrows = 100
    df_test = get_test_data(nrows)
    # ⚠️ in order to push a submission to kaggle you need to use the WHOLE dataset
    # generate_submission_csv(nrows, kaggle_upload=False)
    print(predict(df_test))
