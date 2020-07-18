
#Import Libraries
import pandas as pd
import numpy as np


#Import other files/modules
from config import config
from processing.data_management import load_dataset, save_pipeline
import processing.preprocessors as pp
import pipeline
from predict import make_prediction


def run_training():
    """Train the model"""

    #Read Data
    train = load_dataset(config.TRAIN_FILE)
    #separating SalePrice in Y
    y = np.log(train[config.TARGET])
    train.drop([config.TARGET], axis=1, inplace=True)
    pipeline.price_pipe.fit(train[config.KEEP],y)
    save_pipeline(pipeline_to_save=pipeline.price_pipe)

if __name__=='__main__':
    run_training()
    #Test Prediction
    test_data = load_dataset(file_name=config.TEST_FILE)
    single_test_json = test_data[0:1]

    result = make_prediction(single_test_json)
    print(result)











### Prediction will be done in a separate function call
# pred = pipeline.price_pipe.predict(test[config.KEEP])
# print("Top 10 predictions: ",pred[1:10])

