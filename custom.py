
import pandas as pd
import numpy as np
import joblib
from transformers import CustomImputer, CustomEncoder

import os
import io

g_code_dir = None

import config

def init(code_dir):
    global g_code_dir
    g_code_dir = code_dir


def read_input_data(input_binary_data):
    data = pd.read_csv(io.BytesIO(input_binary_data))
    return data


def transform(data, model):
    """
    Note: This hook may not have to be implemented for your model.
    Modify this method to add data transformation before scoring calls

    Parameters - required in this order, even if unused
    ----------
    data: pd.DataFrame
    model: object, the deserialized model
    
    Returns
    -------
    pd.DataFrame
    """

    # load pre-processing pipeline      
    preprocessing_file = 'preprocessing.pkl'
    preprocessor = joblib.load(os.path.join(g_code_dir, preprocessing_file))

    # apply pre-processing
    preprocessed = preprocessor.transform(data)
    
    return preprocessed


def load_model(code_dir):

    # load model file
    model_file = 'model.pkl'
    model = joblib.load(os.path.join(code_dir, model_file))

    return model
    

def score(data, model, **kwargs):
    """
    Parameters - required in this order, even if unused
    ----------
    data: pd.DataFrame
    model: object, the deserialized model
    """
    # score data
    results = model.predict_proba(data)

    # put predictions into a dataframe
    predictions = pd.DataFrame({'0': results[:, 0], '1':results[:, 1]})

    return predictions


# Adding post_process to obtain binary predictions
def post_process(predictions, model):

    """
    This hook performs required post-processing 

    Parameters
    ----------
    predictions: pd.DataFrame
    model: object, the deserialized model
    
    Returns
    -------
    pd.DataFrame
    """

    # apply prediction threshold to predicted scores
    y_pred = (predictions.iloc[:,1] >= config.PREDICTION_THRESHOLD).astype(int)

    predictions['1'] = y_pred
    predictions['0'] = 1 - predictions['1']

    return predictions       
