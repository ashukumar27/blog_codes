
#Import Libraries
import pandas as pd
import numpy as np


#Import other files/modules
import config
from data_management import load_dataset
import preprocessors as pp
import pipeline


train = load_dataset(config.TRAIN_FILE)
test = load_dataset(config.TEST_FILE)


#separating SalePrice in Y
y = train[config.TARGET]
train.drop([config.TARGET], axis=1, inplace=True)



pipeline.price_pipe.fit(train[config.KEEP],y)
pred = pipeline.price_pipe.predict(test[config.KEEP])

print("Top 10 predictions: ",pred[1:10])




# # # # ## # # # # #      CODE BEFORE PIPELINING - EARLIER VERSION   # # # # # 
### Used in Earlier Code : Multiple FUnctions, to be replaced by Pipeline ###

# #Combine train and test data
# data = pd.concat([train,test], axis=0)

# data = data[config.KEEP].copy()

# #Data Preprocessing functions from preprocessors.py
# data = pp.numerical_imputer(data, config.NUMERICAL_FEATURES)
# data = pp.categorical_imputer(data, config.CATEGORICAL_FEATURES)
# data = pp.rare_label_cat_imputer(data, config.FEATURES_TO_ENCODE)
# data = pp.categorical_encoder(data, config.FEATURES_TO_ENCODE)
# data = pp.temporal_transform(data, config.TEMPORAL_FEATURES, 
#                              config.TEMPORAL_COMPARISON)
# data = pp.log_transform(data, config.LOG_FEATURES)
# data = pp.drop_features(data, config.DROP_FEATURES)

##############################################################################