# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 21:52:17 2019

@author: Mehfuz A Rahman
"""


# Step 1 - Load Data
import pandas as pd
dataset = pd.read_csv("50_Startups.csv")
dataset.dropna()
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values

#Visualization of the dataset
dataset.shape
dataset.info()

import matplotlib.pyplot as plt
plt.style.use('ggplot')

# Histogram of the R&D Spend

dataset.iloc[:,0].plot(kind='hist',color='blue',edgecolor='black',alpha=0.7,figsize=(10,7))
plt.title('Distribution of 50 Startups data', size=24)
plt.xlabel('Amount $', size=18)
plt.ylabel('Frequency', size=18)

# Histogram of the Administration Spend

dataset.iloc[:,1].plot(kind='hist',color='red',edgecolor='black',alpha=0.7,figsize=(10,7))
plt.title('Distribution of 50 Startups data', size=24)
plt.xlabel('Amount $', size=18)
plt.ylabel('Frequency', size=18)

# Histogram of the Marketing Spend

dataset.iloc[:,2].plot(kind='hist',color='green',edgecolor='black',alpha=0.7,figsize=(10,7))
plt.title('Distribution of 50 Startups data', size=24)
plt.xlabel('Amount $', size=18)
plt.ylabel('Frequency', size=18)

# Histogram of the profit

dataset.iloc[:,4].plot(kind='hist',color='yellow',edgecolor='black',alpha=0.7,figsize=(10,7))
plt.title('Distribution of 50 Startups data', size=24)
plt.xlabel('Amount $', size=18)
plt.ylabel('Frequency', size=18)

plt.legend(labels=['R&D Spend','Marketing','Administration','Profit'])


# Step 2 - Encode Categorical Data

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_X = LabelEncoder()
X[:,3] = labelEncoder_X.fit_transform(X[:,3])

oneHotEncoder = OneHotEncoder(categorical_features=[3])
X = oneHotEncoder.fit_transform(X).toarray()

# Step 3 - Dummy Trap

X = X[:,1:]
print(dataset)

# Step 4 - Split Data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)

# Step 5 - Fit Regressor
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Step 6 - Predict
y_pred = regressor.predict(X_test)


# Add ones
import numpy as np
ones = np.ones(shape = (50,1), dtype=int)
X = np.append(arr = ones, values= X, axis=1)


# Backward Elimination
import statsmodels.api as sm
X_opt = X[:,[0,1,2,3,4,5]]
model = sm.OLS(y, X_opt)
results = model.fit()
print(results.summary())

X_opt = X[:,[0,1,3,4,5]]
model = sm.OLS(y, X_opt)
results = model.fit()
print(results.summary())


X_opt = X[:,[0,3,4,5]]
model = sm.OLS(y, X_opt)
results = model.fit()
print(results.summary())

X_opt = X[:,[0,3,5]]
model = sm.OLS(y, X_opt)
results = model.fit()
print(results.summary())


X_opt = X[:,[0,3]]
model = sm.OLS(y, X_opt)
results = model.fit()
print(results.summary())

    


