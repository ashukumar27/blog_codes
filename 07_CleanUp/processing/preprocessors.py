import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

#Local Files
from config import config


#Numerical Imputer
class NumericalImputer(BaseEstimator,TransformerMixin):
    """Numerical Data Missing Value Imputer"""
    def __init__(self, variables=None):
            self.variables = variables
    
    def fit(self, X,y=None):
        self.imputer_dict_={}
        for feature in self.variables:
            self.imputer_dict_[feature] = X[feature].mode()[0]
        return self

    def transform(self,X):
        X=X.copy()
        for feature in self.variables:
            X[feature].fillna(self.imputer_dict_[feature],inplace=True)
        return X




#Categorical Imputer
class CategoricalImputer(BaseEstimator,TransformerMixin):
    def __init__(self, variables=None):
        self.variables = variables
    
    def fit(self, X,y=None):
        self.imputer_dict_={}
        for feature in self.variables:
            self.imputer_dict_[feature] = X[feature].mode()[0]
        return self
    
    def transform(self, X):
        X=X.copy()
        for feature in self.variables:
            X[feature].fillna(self.imputer_dict_[feature],inplace=True)
        return X



#Rare label Categorical Encoder

class RareLabelCategoricalImputer(BaseEstimator,TransformerMixin):
    def __init__(self, tol=0.05, variables=None):
        self.tol=tol
        self.variables=variables
    
    def fit(self, X, y=None):
        self.encoder_dict_={}
        for var in self.variables:
            # the encoder will learn the most frequent categories
            t = pd.Series(X[var].value_counts() / np.float(len(X)))
            # frequent labels:
            self.encoder_dict_[var] = list(t[t >= self.tol].index)
        return self

    def transform(self, X):
        X=X.copy()
        for feature in self.variables:
            X[feature] = np.where(X[feature].isin(self.encoder_dict_[feature]), X[feature], 'Rare')
        return X



#Categorical Encoder
class CategoricalEncoder(BaseEstimator,TransformerMixin):
    def __init__(self, variables=None):
        self.variables=variables
    
    def fit(self, X,y):
        self.encoder_dict_ = {}
        for var in self.variables:
            t = X[var].value_counts().sort_values(ascending=True).index 
            self.encoder_dict_[var] = {k:i for i,k in enumerate(t,0)}
        return self
    
    def transform(self,X):
        X=X.copy()
        ##This part assumes that categorical encoder does not intorduce and NANs
        ##In that case, a check needs to be done and code should break
        for feature in self.variables:
            X[feature] = X[feature].map(self.encoder_dict_[feature])
        return X

# #Temporal Variables
class TemporalVariableEstimator(BaseEstimator,TransformerMixin):
    def __init__(self, variables=None, reference_variable = None):
        self.variables=variables
        self.reference_variable = reference_variable
    
    def fit(self, X,y=None):
        #No need to put anything, needed for Sklearn Pipeline
        return self
    
    def transform(self, X):
        X=X.copy()
        for var in self.variables:
            X[var] = X[var]-X[self.reference_variable]
        return X 



    
# # Log Transformations
class LogTransformation(BaseEstimator, TransformerMixin):
    def __init__(self, variables=None):
        self.variables = variables
    
    def fit(self, X,y):
        return self

    ### Need to check in advance if the features are all non negative >0
    ### If yes, needs to be transformed properly
    def transform(self,X):
        X=X.copy()
        for var in self.variables:
            X[var] = np.log(X[var])
        return X


# # Drop Features
class DropFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, variables_to_drop=None):
        self.variables_to_drop = variables_to_drop
    
    def fit(self, X,y=None):
        return self 
    
    def transform(self, X):
        X=X.copy()
        X= X.drop(self.variables_to_drop, axis=1)
        return X
    
    