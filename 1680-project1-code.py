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

#import ACS data set
ddi_codebook = readers.read_ipums_ddi(r"C:\Users\baron\Downloads\usa_00004.xml")
ipums_df = readers.read_microdata(ddi_codebook, r"C:\Users\baron\Downloads\usa_00004.dat.gz")

