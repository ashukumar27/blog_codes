#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 12:57:42 2020

@author: ashutosh.k

"""
#Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Lasso
from sklearn import metrics


#Import other files/modules
import config
from data_management import load_dataset
import preprocessors as pp


train = load_dataset(config.TRAIN_FILE)
test = load_dataset(config.TEST_FILE)


#separating SalePrice in Y
y = train[config.TARGET]
train.drop([config.TARGET], axis=1, inplace=True)

#Combine train and test data
data = pd.concat([train,test], axis=0)

data = data[config.KEEP].copy()

#Data Preprocessing functions from preprocessors.py
data = pp.numerical_imputer(data, config.NUMERICAL_FEATURES)
data = pp.categorical_imputer(data, config.CATEGORICAL_FEATURES)
data = pp.rare_label_cat_imputer(data, config.FEATURES_TO_ENCODE)
data = pp.categorical_encoder(data, config.FEATURES_TO_ENCODE)
data = pp.temporal_transform(data, config.TEMPORAL_FEATURES, 
                             config.TEMPORAL_COMPARISON)
data = pp.log_transform(data, config.LOG_FEATURES)
data = pp.drop_features(data, config.DROP_FEATURES)

##############################################################################


## Split Train and Test

train_clean = data.iloc[:train.shape[0],:]
test_clean = data.iloc[train.shape[0]:,:]

#Split Train and Test
X_train, X_test, y_train, y_test = train_test_split(train_clean, y, 
                                                    random_state=42, test_size=0.15)
y_train = np.log(y_train)
y_test = np.log(y_test)

# Run Model
model = Lasso(alpha=0.005, random_state=0)
model.fit(X_train,y_train)
pred = model.predict(X_test)

# Model Evaluation
#MSE
print("MSE : ",metrics.mean_squared_error(pred, y_test))
#MAE
print("MAE : ",metrics.mean_absolute_error(pred, y_test))
#RMSE
print("RMSE : ",np.sqrt(metrics.mean_squared_error(pred, y_test)))
#R2
print("R-sq : ",metrics.r2_score(pred, y_test))


# Prediction on actual Test Data
#test_clean is the transformed original test data; x_test is the 15% split from training data, 
#apologies for similar names
pred_test = np.exp(model.predict(test_clean))

print("Top 10 predictions: ",pred_test[1:10])


