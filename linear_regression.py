### Tools for linear regression ###

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm


def simulate_data(nobs):
    """
    Simulates data for testing linear_regression models.
    INPUT
        nobs (int) the number of observations in the dataset
    RETURNS
        data (dict) contains X, y, and beta vectors.
    """
    set.seed(1)
    beta = 1/9000
    x1 = np.random.exponential(beta, nobs)
    lambda_param = 1/15
    x2 = np.random.poisson(lambda_param, nobs) #Needs to be a matrix nvm
    betas = np.random.random(2)
    epsilon = np.random.random(nobs)
    #Combining the predictor variables into X: 
    x1, x2 = x1.reshape((nobs, 1)), x2.reshape((nobs,1))
    X = np.concatenate((x1,x2), axis = 1)
    y = np.dot(X, betas) + epsilon
    val_dict = { "X": X, "beta": betas, "y": y}
    return(val_dict)


def compare_models(X,y):
    """
    Compares output from different implementations of OLS.
    INPUT
        X (ndarray) the independent variables in matrix form
        y (array) the response variables vector
    RETURNS
        results (pandas.DataFrame) of estimated beta coefficients
    """
    ones_col = np.ones((X.shape[0],1))
    X_with_ones = np.hstack([ones_col, X])
    
    beta = np.linalg.lstsq(X_with_ones, y)[0]
    beta = pd.DataFrame(beta[0], columns = ["Betas"])
    return(beta)

def load_hospital_data(path_to_data):
    """
    Loads the hospital charges data set found at data.gov.
    INPUT
        path_to_data (str) indicates the filepath to the hospital charge data (csv)
    RETURNS
        clean_df (pandas.DataFrame) containing the cleaned and formatted dataset for regression
    """
    df = pd.read_csv(path_to_data)
    df_by_provider = pd.groupby('Provider Id').sum()
    return df_by_provider



def prepare_data():
    """
    Prepares hospital data for regression (basically turns df into X and y).
    INPUT
        df (pandas.DataFrame) the hospital dataset
    RETURNS
        data (dict) containing X design matrix and y response variable
    """

   


def run_hospital_regression():
    """
    Loads hospital charge data and runs OLS on it.
    INPUT
        path_to_data (str) filepath of the csv file
    RETURNS
        results (str) the statsmodels regression output
    """
    pass
 

### END ###

