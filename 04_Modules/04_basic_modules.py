#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 12:51:11 2020

@author: ashutosh.k

Code into Modules
"""

#Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Lasso
from sklearn import metrics

import config


## Read Data
train = pd.read_csv(config.DATAPATH+config.TRAIN_FILE)
test = pd.read_csv(config.DATAPATH+config.TEST_FILE)

#separating SalePrice in Y
y = train[config.TARGET]
train.drop([config.TARGET], axis=1, inplace=True)

#Combine train and test data
data = pd.concat([train,test], axis=0)

data = data[config.KEEP].copy()

#Numerical Imputer
for var in config.NUMERICAL_FEATURES:
    data[var].fillna(data[var].mode()[0], inplace=True)


#Categorical Imputer
for var in config.CATEGORICAL_FEATURES:
    data[var].fillna(data[var].mode()[0], inplace=True)

#Rare label Categorical Imputer
encoder_dict_ = {}
tol=0.05

for var in config.FEATURES_TO_ENCODE:
    # the encoder will learn the most frequent categories
    t = pd.Series(data[var].value_counts() / np.float(len(data)))
    # frequent labels:
    encoder_dict_[var] = list(t[t >= tol].index)
    
for var in config.FEATURES_TO_ENCODE:
    data[var] = np.where(data[var].isin(
                encoder_dict_[var]), data[var], 'Rare')
    
#Categorical Imputer
encoder_dict_ ={}
for var in config.FEATURES_TO_ENCODE:
    t = data[var].value_counts().sort_values(ascending=True).index  #Sorting on freq, should be done on target, just saving some time here
    encoder_dict_[var] = {k:i for i,k in enumerate(t,0)}
    
## Mapping using the encoder dictionary
for var in config.FEATURES_TO_ENCODE:
    data[var] = data[var].map(encoder_dict_[var])
    

#Temporal Variables
for var in config.TEMPORAL_FEATURES:
    data[var] = data[var]-data[config.TEMPORAL_COMPARISON]
    
# Log Transformations
for var in config.LOG_FEATURES:
    data[var] = np.log(data[var])
    
# Drop Features
data.drop(config.DROP_FEATURES, axis=1, inplace=True)

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
