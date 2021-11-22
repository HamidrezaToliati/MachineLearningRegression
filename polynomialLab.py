#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl


# In[5]:


df = pd.read_csv("FUELCONSUMPTION.csv")
df.head()


# In[6]:


cdf = df[['ENGINESIZE', 'CYLINDERS' ,'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]
cdf.head()


# In[7]:


plt.scatter(cdf['ENGINESIZE'], cdf['CO2EMISSIONS'])
plt.xlabel('engine size')
plt.ylabel('CO2 emissions')
plt.show()


# In[15]:


msk = np.random.rand( len(cdf) ) < 0.8
train_set = cdf[msk]
test_set = cdf[~msk]

plt.scatter(train_set['ENGINESIZE'], train_set['CO2EMISSIONS'])
plt.scatter(test_set['ENGINESIZE'], test_set['CO2EMISSIONS'], color = 'red')
plt.xlabel('cylinders')
plt.ylabel('CO2 emissions')
plt.show()


# In[35]:


from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

train_x = np.asanyarray(train_set[['ENGINESIZE']])
train_y = np.asanyarray(train_set[['CO2EMISSIONS']])

test_x = np.asanyarray(test_set[['ENGINESIZE']])
test_y = np.asanyarray(test_set[['CO2EMISSIONS']])

poly = PolynomialFeatures(degree = 2)
train_x_poly = poly.fit_transform(train_x)
print(train_x_poly)


# In[39]:


regr = linear_model.LinearRegression()
regr.fit(train_x_poly, train_y)
print("coef is: ", regr.coef_)
print("intercept is: ", regr.intercept_)


# In[51]:


XX = np.arange(0, 10, 0.1)
YY = regr.intercept_[0] + regr.coef_[0][1]*XX + regr.coef_[0][2]*np.power(XX, 2)
plt.scatter(cdf['ENGINESIZE'], cdf['CO2EMISSIONS'])
plt.plot(XX, YY, '--r')
plt.xlabel('engine size')
plt.ylabel('CO2 emissions')
plt.show()


# In[49]:


from sklearn.metrics import r2_score
test_x_poly = poly.fit_transform(test_x)
eval_y = regr.predict(test_x_poly)
R2 = r2_score(test_y, eval_y)
print("the R2 score is: %.2f" % R2)


# # practice
# #### test for cubic instead

# In[56]:


from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
train_x = np.asanyarray(train_set[['ENGINESIZE']])
train_y = np.asanyarray(train_set[['CO2EMISSIONS']])

test_x = np.asanyarray(test_set[['ENGINESIZE']])
test_y = np.asanyarray(test_set[['CO2EMISSIONS']])

poly3 = PolynomialFeatures(degree = 3)
train_x_poly = poly3.fit_transform(train_x)


# In[58]:


regr3 = linear_model.LinearRegression()
regr3.fit(train_x_poly, train_y)

print("coef3 is: ", regr3.coef_)
print("intercept3 is: ", regr3.intercept_)


# In[60]:


XX3 = np.arange(0, 10, 0.1)
YY3 = regr3.intercept_[0] + regr3.coef_[0][1]*XX3 + regr3.coef_[0][2]*np.power(XX3, 2) + regr3.coef_[0][3]*np.power(XX3, 3)
plt.scatter(cdf['ENGINESIZE'], cdf['CO2EMISSIONS'])
plt.plot(XX3, YY3, '--r')
plt.xlabel('engine size')
plt.ylabel('CO2 emissions')
plt.show()


# In[63]:


from sklearn.metrics import r2_score
test_x_poly3 = poly3.fit_transform(test_x)
eval_y = regr3.predict(test_x_poly3)
R2 = r2_score(test_y, eval_y)

print("R2 score is: %.2f" % R2)

