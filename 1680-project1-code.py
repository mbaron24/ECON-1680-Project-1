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

#DATA CLEANING

#reducing data set to only include households that are eligible for snap benefits
ipums_df = ipums_df[((ipums_df['FAMSIZE'] == 1) & (ipums_df['HHINCOME'] <= 1987))\
                    + ((ipums_df['FAMSIZE'] == 2) & (ipums_df['HHINCOME'] <= 2686))
                        + ((ipums_df['FAMSIZE'] == 3) & (ipums_df['HHINCOME'] <= 3386))\
                            + ((ipums_df['FAMSIZE'] == 4) & (ipums_df['HHINCOME'] <= 4087))\
                                + ((ipums_df['FAMSIZE'] == 5) & (ipums_df['HHINCOME'] <= 4786))\
                                    + ((ipums_df['FAMSIZE'] >= 6) & (ipums_df['HHINCOME'] <= 5485))]

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

