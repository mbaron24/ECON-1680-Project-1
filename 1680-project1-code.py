# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 11:05:02 2023

@author: baron
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import statsmodels.api as sm
import ipumspy 
from statsmodels.iolib.summary2 import summary_col

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, LassoCV
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import normalize
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from ipumspy import readers, ddi

#IMPORT DATA SET
ddi_codebook = readers.read_ipums_ddi(r"C:\Users\baron\Documents\GitHub\ECON-1680-Project-1\ACS-RI-snap.xml")
ipums_df = readers.read_microdata(ddi_codebook, r"C:\Users\baron\Documents\GitHub\ECON-1680-Project-1\ACS-RI-snap.dat.gz")


#DATA CLEANING AND NORMALIZATION

#reducing data set to only include households that are eligible for snap benefits
ipums_df = ipums_df[((ipums_df['FAMSIZE'] == 1) & (ipums_df['HHINCOME']/12 <= 1987))\
                    + ((ipums_df['FAMSIZE'] == 2) & (ipums_df['HHINCOME']/12 <= 2686))
                        + ((ipums_df['FAMSIZE'] == 3) & (ipums_df['HHINCOME']/12 <= 3386))\
                            + ((ipums_df['FAMSIZE'] == 4) & (ipums_df['HHINCOME']/12 <= 4087))\
                                + ((ipums_df['FAMSIZE'] == 5) & (ipums_df['HHINCOME']/12 <= 4786))\
                                    + ((ipums_df['FAMSIZE'] == 6) & (ipums_df['HHINCOME']/12 <= 5485))\
                                        + ((ipums_df['FAMSIZE'] == 7) & (ipums_df['HHINCOME']/12 <= 6186))\
                                            + ((ipums_df['FAMSIZE'] == 8) & (ipums_df['HHINCOME']/12 <= 6887))\
                                                + ((ipums_df['FAMSIZE'] == 9) & (ipums_df['HHINCOME']/12 <= 7588))\
                                                    + ((ipums_df['FAMSIZE'] == 10) & (ipums_df['HHINCOME']/12 <= 8289))\
                                                        + ((ipums_df['FAMSIZE'] >= 11) & (ipums_df['HHINCOME']/12 <= 8990))]

#re-index dataframe
ipums_df.reindex()

#creating one dummy variable to represent if observed person has a disability
ipums_df['DISSABILITY'] = ipums_df['DIFFREM'].apply(lambda x: 1 if x == 2 else 0)
ipums_df['DISSABILITY'] = ipums_df['DIFFPHYS'].apply(lambda x: 1 if x == 2 else 0)
ipums_df['DISSABILITY'] = ipums_df['DIFFMOB'].apply(lambda x: 1 if x == 2 else 0)
ipums_df['DISSABILITY'] = ipums_df['DIFFCARE'].apply(lambda x: 1 if x == 2 else 0)
ipums_df['DISSABILITY'] = ipums_df['DIFFSENS'].apply(lambda x: 1 if x == 2 else 0)
ipums_df['DISSABILITY'] = ipums_df['DIFFEYE'].apply(lambda x: 1 if x == 2 else 0)
ipums_df['DISSABILITY'] = ipums_df['DIFFHEAR'].apply(lambda x: 1 if x == 2 else 0)

#create dummy variable for english speaking
ipums_df['SPEAKENG'] = ipums_df['SPEAKENG'].apply(lambda x: 0 if x == 1 or x == 0 else 1)

#create veteran status dummy variable
ipums_df['VETSTAT'] = ipums_df['VETSTAT'].apply(lambda x: 1 if x == 2 else 0)

#create employment dummy variable
ipums_df['EMPSTAT'] = ipums_df['EMPSTAT'].apply(lambda x: 1 if x == 1 else 0)

#create dummy variable for Hispanic
ipums_df['HISPAN-binary'] = ipums_df['HISPAN'].apply(lambda x: 0 if x == 0 or x==9 else 1)

#normalize variables and rename columns
df_ipums_scaled = normalize(ipums_df)
df_ipums_scaled = pd.DataFrame(df_ipums_scaled, columns=ipums_df.columns, index=ipums_df.index) 

#create an interaction variable for famsize and metro status
df_ipums_scaled['FAMSIZExMETRO'] = df_ipums_scaled['FAMSIZE']*df_ipums_scaled['METRO']

#create an interaction variable for sex and race
df_ipums_scaled['SEXxRACE'] = df_ipums_scaled['RACE']*df_ipums_scaled['SEX']

#create an interaction variable for race and age
df_ipums_scaled['AGExRACE'] = df_ipums_scaled['RACE']*df_ipums_scaled['AGE']

#create an interaction variable for sex and age
df_ipums_scaled['SEXxAGE'] = df_ipums_scaled['AGE']*df_ipums_scaled['SEX']

#create an interaction variable for sex and age
df_ipums_scaled['SEXxAGE'] = df_ipums_scaled['AGE']*df_ipums_scaled['SEX']

#create an interaction variable for sex, race, and age
df_ipums_scaled['SEXxAGExRACE'] = df_ipums_scaled['AGE']*df_ipums_scaled['SEX']*df_ipums_scaled['RACE']

#drop variables that are not important to the research question
df_ipums_scaled.dropna(subset=['REGION', 'METRO', 'HOMELAND', 'HHINCOME', 'FAMSIZE', 'SEX', 'AGE', \
                    'MARST', 'RACE', 'HISPAN-binary', 'CITIZEN', 'YRSUSA1', 'SPEAKENG', 'EDUC', \
                        'EMPSTAT','VETSTAT','DISSABILITY','FAMSIZExMETRO','SEXxRACE', \
                            'AGExRACE', 'SEXxAGE', 'SEXxAGExRACE', 'FOODSTMP'],inplace=True)

#set output variable 
y = df_ipums_scaled['FOODSTMP']

#set regressors 
X = df_ipums_scaled[['REGION', 'METRO', 'HOMELAND', 'HHINCOME', 'FAMSIZE', 'SEX', 'AGE', \
                    'MARST', 'RACE', 'HISPAN-binary', 'CITIZEN', 'YRSUSA1', 'SPEAKENG', 'EDUC', \
                        'EMPSTAT','VETSTAT','DISSABILITY','FAMSIZExMETRO','SEXxRACE',\
                            'AGExRACE', 'SEXxAGE', 'SEXxAGExRACE']]
    
#splits the data into training and testing groups
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1680)

scaler=StandardScaler() #removes mean and scales to unit variance
scaler.fit(X_train) #trains the scaling function using the training data
X_train = scaler.transform(X_train) #normalizes training set X using scaler
X_test = scaler.transform(X_test) #normalizes test set X using scaler
    

#OLS REGRESSION

#OLS on training data
olsReg = sm.OLS(y_train,X_train).fit()
df_results = pd.DataFrame({'Variable':X.columns, 'Coeff OLS':olsReg.params})

#calculate mean squared error in order to estimate out of sample accuracy
ols_y_pred = olsReg.predict(X_test)
OLS_MSE = mean_squared_error(y_test, ols_y_pred)


#RIDGE REGRESSION

#find best alpha value
# Construct vector of alpha values
alphas = np.linspace(0.01, 5, num=50)

# Construct vectors to store mean prediction errors and coefficients
cv_errs = []
coefs = []
MSE = 10
bestalpha = 0

# Loop for running ridge regression for different values of alpha
for a in alphas:
    
    # define pipeline object
    ridgeReg = make_pipeline(StandardScaler(with_mean=False), Ridge(alpha= a * X_train.shape[0]))
    # run Ridge regression
    ridgeReg.fit(X_train, y_train)
    # obtain predicted values of output
    y_pred = ridgeReg.predict(X_test)
    # compute mean squared error
    cv_errs.append(mean_squared_error(y_test, y_pred))
    # store coefficients
    coefs.append(ridgeReg['ridge'].coef_)
    
    # store value of alpha that minimizes the mean squared error
    if mean_squared_error(y_test, y_pred) < MSE: #np.mean((y_pred - y_test)**2)
        MSE = mean_squared_error(y_test, y_pred)
        bestalpha_ridge = a


#run ridge regression using best alpha 
alpha = bestalpha_ridge
ridgeReg = make_pipeline(StandardScaler(with_mean=False), Ridge(alpha= alpha * X_train.shape[0]))
ridgeReg.fit(X_train, y_train)

#calculate mean squared error in order to estimate out of sample accuracy
ridge_y_pred = ridgeReg.predict(X_test)
RIDGE_MSE = mean_squared_error(y_test, ridge_y_pred)

# Add coefficients to results dataframe
df_results['Coeff RIDGE'] = ridgeReg['ridge'].coef_


#LASSO REGRESSION

# Define model
lassoReg = make_pipeline(StandardScaler(with_mean=False), Lasso())

# Define parameter grid to search over using grid search
alphas = np.linspace(1e-14, 1, num=50)*np.sqrt(X_train.shape[0])
params = {'lasso__alpha' : alphas}

# Set up the grid search
gsLasso = GridSearchCV(lassoReg, params, n_jobs=-1, cv=10)

# Fit gs to data
gsLasso.fit(X, y)

# Check best alpha
bestalpha_lasso = list(gsLasso.best_params_.values())[0] / np.sqrt(X_train.shape[0])

# Run lasso regression
lassoReg = make_pipeline(StandardScaler(with_mean=False), Lasso(alpha= bestalpha_lasso))
lassoReg.fit(X_train, y_train)

#calculate mean squared error in order to estimate out of sample accuracy
lasso_y_pred = lassoReg.predict(X_test)
LASSO_MSE = mean_squared_error(y_test, lasso_y_pred)

# Add coefficients to dataframe
df_results['Coeff LASSO'] = lassoReg['lasso'].coef_
print(df_results)

#see which variables are most predictive of SNAP participation
lasso_best_vars = df_results[(df_results['Coeff LASSO'] > 0) | \
                             (df_results['Coeff LASSO'] < 0)].sort_values("Coeff LASSO", ascending=False, key=abs)
lasso_best_vars_list = lasso_best_vars["Variable"].tolist()
print(lasso_best_vars_list)


#NEURAL NETWORK
