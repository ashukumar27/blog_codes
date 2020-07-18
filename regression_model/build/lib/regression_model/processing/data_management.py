from regression_model.config import config
import pandas as pd
import joblib

def load_dataset(file_name):
    _data = pd.read_csv(config.DATAPATH + file_name)
    return _data

def save_pipeline(pipeline_to_save):
    save_file_name = 'lasso_regression_v1.pkl'
    save_path = config.SAVED_MODEL_PATH+save_file_name
    joblib.dump(pipeline_to_save, save_path)
    print("Saved Pipeline : ",save_file_name)


def load_pipeline(pipeline_to_load):
    save_path = config.SAVED_MODEL_PATH
    trained_model = joblib.load(save_path+pipeline_to_load)
    return trained_model