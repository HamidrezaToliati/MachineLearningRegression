#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[4]:


df = pd.read_csv("FuelConsumption.csv")
df.head()
df.describe()


# In[5]:


cdf = df[["ENGINESIZE", 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]
cdf.head()


# In[6]:


viz = cdf[['CYLINDERS', 'ENGINESIZE', 'CO2EMISSIONS', 'FUELCONSUMPTION_COMB']]
# viz.hist()
fig, axs = plt.subplots(2, 2, tight_layout=True)
plt.subplot(221)
plt.hist(cdf['CYLINDERS'])
plt.xlabel('cylinders')
plt.subplot(222)
plt.hist(cdf['ENGINESIZE'])
plt.xlabel('engine size')
plt.subplot(223)
plt.hist(cdf['CO2EMISSIONS'])
plt.xlabel('CO2 emissions')
plt.subplot(224)
plt.hist(cdf['FUELCONSUMPTION_COMB'])
plt.xlabel('fuel consumption comb')
plt.show()


# In[7]:


figure, axs = plt.subplots(2, 2, tight_layout=True, sharey=True)
plt.subplot(221)
plt.scatter(cdf['FUELCONSUMPTION_COMB'], cdf['CO2EMISSIONS'], color='blue')
plt.xlabel('fuel consumption comb')
plt.ylabel('CO2 emissions')
plt.subplot(222)
plt.scatter(cdf['CYLINDERS'], cdf['CO2EMISSIONS'], color='blue')
plt.xlabel('cylinders')
# plt.ylabel('CO2 emissions')
plt.subplot(223)
plt.scatter(cdf['ENGINESIZE'], cdf['CO2EMISSIONS'], color='blue')
plt.xlabel('engine size')
plt.ylabel('CO2 emissions')
plt.show()


# In[9]:


msk = np.random.rand(len(cdf)) < 0.8
train = cdf[msk]
test = cdf[~msk]


# In[10]:


plt.scatter(train['ENGINESIZE'], train['CO2EMISSIONS'], color='blue')
plt.scatter(test['ENGINESIZE'], test['CO2EMISSIONS'], color='red')
plt.xlabel('engine size')
plt.ylabel('CO2 emissions')
plt.show()


# In[12]:


from sklearn import linear_model
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(train_x, train_y)
a = regr.coef_[0][0]
b = regr.intercept_[0]
print("coefficients: ", regr.coef_)
print("intercept: ", regr.intercept_)


# In[13]:


plt.scatter(train['ENGINESIZE'], train['CO2EMISSIONS'], color='blue')
# plt.scatter(test['ENGINESIZE'], test['CO2EMISSIONS'], color='yellow')
plt.plot(train_x, a*train_x + b, color='red')
plt.xlabel('engine size')
plt.ylabel('CO2 emissions')
plt.show()


# In[16]:


from sklearn.metrics import r2_score
test_x = test[['ENGINESIZE']]
test_y = test[['CO2EMISSIONS']]
eval_y = regr.predict(test_x)

print("mean absolute error is: %.2f" % np.mean(np.absolute(eval_y - test_y)))
print("residual sum of squares: %.2f" % np.mean((eval_y - test_y)**2))
print("R2_score: %.2f" % r2_score(test_y, eval_y))

