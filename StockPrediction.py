#!/usr/bin/env python
# coding: utf-8

# In[238]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,r2_score,confusion_matrix
get_ipython().run_line_magic('matplotlib', 'inline')


# In[239]:


dataset=pd.read_csv('NKE.csv')


# In[240]:


dataset.shape


# In[241]:


dataset.head()


# In[242]:


dataset.isnull().any()


# In[243]:


dataset.hist(bins=50,figsize=(20,15),color='green')
plt.show()


# In[244]:


x=dataset['Open']
y=dataset['Close']
plt.plot(x,label='Open')
plt.plot(y,label='Close')
m=dataset['Adj_High']
n=dataset['Adj_Low']
plt.plot(m,label='Adj_High')
plt.plot(n,label='Adj_Low')
plt.legend()
plt.show()


# In[245]:


sns.distplot(dataset['High'],kde=True,bins=10)
sns.distplot(dataset['Low'],kde=True,bins=10)#both the kde's ovelapped hence teh aveage of both High  and Low lie in similar range


# In[246]:


m=dataset['Adj_High']
n=dataset['High']
plt.plot(m,label='Adj_High')
plt.plot(n,label='High')
plt.legend()
plt.show()


# In[247]:


m=dataset['Adj_Low']
n=dataset['Low']
plt.plot(m,label='Adj_Low')
plt.plot(n,label='Low')
plt.legend()
plt.show()


# In[248]:


forecast=int(input())


# In[249]:


dataset=dataset[['Adj_Close']]
print(dataset.head())


# In[250]:


dataset['Prediction']= dataset[['Adj_Close']].shift(-forecast)


# In[251]:


print(dataset.tail())


# In[252]:


X = np.array(dataset.drop(['Prediction'],1))


# In[253]:


dataset.describe()


# In[254]:


X = X[:-forecast]


# In[255]:


Y = np.array(dataset['Prediction'])


# In[256]:


Y = Y[:-forecast]


# In[257]:


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)


# In[258]:


svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_rbf.fit(x_train, y_train)

svm_confidence = svr_rbf.score(x_test, y_test)
print("svm confidence: ", svm_confidence*100,"%")


# In[259]:


linear_reg = LinearRegression()
# Train the model
linear_reg.fit(x_train, y_train)

linear_confidence = linear_reg.score(x_test, y_test)
print("linear regression  confidence: ", linear_confidence)


# In[260]:


x_forecast = np.array(dataset.drop(['Prediction'],1))[-forecast:]

print(x_forecast)


# In[261]:


linear_reg_prediction = linear_reg.predict(x_forecast)
print('Linear Model Prediction is :',linear_reg_prediction)


# In[262]:


svm_prediction = svr_rbf.predict(x_forecast)
print("SVM MODEL prediction is :")
print(svm_prediction)


# In[263]:


m=linear_reg_prediction
n=svm_prediction
plt.plot(m,label='Linear Regession Prediction')
plt.plot(n,label='SVM prediction')
plt.legend()
plt.show()


# In[ ]:





# In[ ]:




