import pathlib
import os


DATAPATH = "../data/HousingPrediction/"
SAVED_MODEL_PATH = "../HousingPriceAllEvoution/07_CleanUp/trained_models/"
## These datapaths need to be changed in the final package

TRAIN_FILE = 'train.csv'
TEST_FILE = 'test.csv'

TARGET = 'SalePrice'
## Features to keep
KEEP = ['MSSubClass', 'MSZoning', 'Neighborhood',
            'OverallQual', 'OverallCond', 'YearRemodAdd',
            'RoofStyle', 'MasVnrType', 'BsmtQual', 'BsmtExposure',
            'HeatingQC', 'CentralAir', '1stFlrSF', 'GrLivArea',
            'BsmtFullBath', 'KitchenQual', 'Fireplaces', 'FireplaceQu',
            'GarageType', 'GarageFinish', 'GarageCars', 'PavedDrive',
            'LotFrontage','YrSold'] #Final feature to keep in data

NUMERICAL_FEATURES = ['LotFrontage'] #Numerical
CATEGORICAL_FEATURES = ['MasVnrType', 'BsmtQual', 'BsmtExposure','FireplaceQu', 
                'GarageCars','GarageType', 'GarageFinish','MSZoning','BsmtFullBath',
                'KitchenQual'] #Categorical

FEATURES_TO_ENCODE = ['MSZoning', 'Neighborhood', 'RoofStyle', 'MasVnrType','BsmtQual', 
                      'BsmtExposure', 'HeatingQC', 'CentralAir','KitchenQual', 'FireplaceQu', 
                      'GarageType', 'GarageFinish','PavedDrive'] #Features to Encode

TEMPORAL_FEATURES = ['YearRemodAdd']
TEMPORAL_COMPARISON = 'YrSold'

LOG_FEATURES = ['LotFrontage', '1stFlrSF', 'GrLivArea'] #Features for Log Transform

DROP_FEATURES = ['YrSold'] #Features to Drop

