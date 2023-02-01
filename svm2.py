# -*- coding: utf-8 -*-
"""
Created on Sat Jan  1 13:17:33 2022

@author: csjh9
"""

from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
  
# For data manipulation
import pandas as pd
import numpy as np
  
# To plot
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')
  
# To ignore warnings
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('Stock market data.csv')
df

df = df[['Adj Close']]

#Create one more column Prediction shifted 15 days up. 
df['Prediction'] = df[['Adj Close']].shift(-15)
#print data set
print(df)

#Create a data set X and convert it into numpy array , which will be having actual values
X = np.array(df.drop(['Prediction'],1))
#Remove the last 15 rows
X = X[:-15]
print(X)

# Create a dataset y which will be having Predicted values and convert into numpy array
y = np.array(df['Prediction'])
# Remove Last 15 rows
y = y[:-15]
print(y)

# Split the data into train and test with 90 & 10 % respectively
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# SVM Model
svr = SVR(kernel='rbf', C=1e3, gamma=0.1)
# Train the model 
svr.fit(x_train, y_train)

svm_confidence = svr.score(x_test, y_test)
print("svm confidence: ", svm_confidence)

forecast = np.array(df.drop(['Prediction'],1))[-15:]
print(forecast)

# support vector model predictions for the next ‘15’ days
svm_prediction = svr.predict(forecast)
print(svm_prediction)