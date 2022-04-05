"""
These are custrom classes that were built for:

   1.- imputation
   2.- one-hot encoding that returns a dataframe instead of a sparse array

"""

# to create customised preprocessing pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# categorical feature encoding
from sklearn.preprocessing import OneHotEncoder

# to perform imputation
from sklearn.impute import SimpleImputer

# to handle datasets
import pandas as pd
import numpy as np

##################################################################


class CustomImputer(BaseEstimator, TransformerMixin):
    
    def __init__(self, ordinal_vars, nominal_vars):
        
        # lists of ordinal and nominal variables
        self.ordinal = ordinal_vars
        self.nominal = nominal_vars
        
        return None
    
    def fit(self, X):
        return self
    
    def transform(self, X):
        
        # replace blanks with nan
        X = X.replace(r'^\s*$', np.nan, regex=True)
        
        # typecast ordinal features to int
        X[self.ordinal] = X[self.ordinal].astype(float).astype(pd.Int32Dtype())
        
        # instantiate imputers
        ordinal_imputer = SimpleImputer(strategy='constant', fill_value=-1)
        nominal_imputer = SimpleImputer(strategy='constant', fill_value='Missing')
        
        # missing value imputation
        X[self.nominal] = nominal_imputer.fit_transform(X[self.nominal])
        X[self.ordinal] = ordinal_imputer.fit_transform(X[self.ordinal])
                 
        return X    


class CustomEncoder(BaseEstimator, TransformerMixin):   
    
    def __init__(self):
        
        # instantiate one-hot encoder, unknown labels will be ignored
        self.encoder = OneHotEncoder(handle_unknown='ignore')
        
        return None
    
    def fit(self, X):
        
        # fit one-hot encoder
        self.encoder.fit(X) 
        
        return self
    
    def transform(self, X):
        
        # perform one-hot encoding
        X = self.encoder.transform(X)
        
        # transforms encoded sparse array into a dataframe
        df = pd.DataFrame.sparse.from_spmatrix(X)   
            
        return df  
