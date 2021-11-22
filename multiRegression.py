#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl


# In[5]:


df = pd.read_csv("FuelConsumption.csv")
df.head()


# In[9]:


cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]
cdf.head()


# In[10]:


msk = np.random.rand( len(cdf) ) < 0.8
train_set = cdf[msk]
test_set = cdf[~msk]


# In[27]:


fig, axs = plt.subplots(2, 3, tight_layout=True, sharey=True)
plt.subplot(231)
plt.scatter(cdf['ENGINESIZE'], cdf['CO2EMISSIONS'], color='blue')
plt.xlabel('engine size')
plt.subplot(232)
plt.scatter(cdf['CYLINDERS'], cdf['CO2EMISSIONS'], color='blue')
plt.xlabel('cylinders')
plt.subplot(233)
plt.scatter(cdf['FUELCONSUMPTION_CITY'], cdf['CO2EMISSIONS'], color='blue')
plt.xlabel('city consumption')
plt.subplot(234)
plt.scatter(cdf['FUELCONSUMPTION_HWY'], cdf['CO2EMISSIONS'], color='blue')
plt.xlabel('highway consumption')
plt.subplot(235)
plt.scatter(cdf['FUELCONSUMPTION_COMB'], cdf['CO2EMISSIONS'], color='blue')
plt.xlabel('combined consumption')
plt.ylabel('CO2 emissions')

plt.show()


# In[50]:


from sklearn import linear_model
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train_set[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']])
train_y = np.asanyarray(train_set['CO2EMISSIONS'])
regr.fit(train_x, train_y)
print('coef is ', regr.coef_)
print('intercept is, ', regr.intercept_)


# In[52]:


from sklearn.metrics import r2_score
test_x = np.asanyarray(test_set[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']])
test_y = np.asanyarray(test_set['CO2EMISSIONS'])
eval_y = regr.predict(test_x)
R2 = r2_score(test_y, eval_y)
print('R2 score is: %.2f' % R2)

