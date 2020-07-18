import numpy as np
import pandas as pd 

from config import config
from processing.data_management import load_pipeline

pipeline_file_name = 'lasso_regression_v1.pkl'

_price_pipe = load_pipeline(pipeline_file_name)

def make_prediction(input_data):
    data = pd.DataFrame(input_data)
    prediction = _price_pipe.predict(data[config.KEEP])
    output = np.exp(prediction)

    results = {
        'prediction': output,
        'model_name': pipeline_file_name,
        'version':'version1'
    }

    return results

