#!/usr/bin/env python
# coding: utf-8

# # Problem Statement

# A chinese automobail company Geely Auto aspires to enter the US market by setting up their manufacturing unit there and producing cars locally to give competition to their US and European counterparts.
# 
# They have contracted an automobile consulting company to understand the facors on which the pricing of cars depends. specifically, they want to understand the factors affecting the pricing of cars in the American market, since those may be very different from the chinese market. The company wants to know:
# 
# . Which variables are significant in predicting the price of a car
# . How well those variables describe the price of a car

# # Business Goal

# We are required to model the price of cars with the available independent variables. It will be used by the management to understand how exactly the prices vary with the independent variables. They can accordingly manipulate the design of the cars, the business strategy etc. to meet certain price levels, Further, the model will be a good way for management to understand the pricing dynamics of a new market,

# In[105]:


# Import the needfull libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


# To ignore warnings
import warnings
warnings.filterwarnings("ignore")


# In[5]:


# Import the data
data = pd.read_csv('CarPrice_Assignment-Copy1.csv')


# In[6]:


data.head()


# In[12]:


data.shape


# In[13]:


data.info()


# In[14]:


# check the nul value
data.isna().sum()


# In[15]:


#data mean value
data.isnull().mean()


# In[17]:


data['cylindernumber']


# In[24]:


# Here Target variable is a price(Analyzing the target variable)
data['price'].describe()


# In[25]:


# Treating an outlier
sns.boxplot(data['price'], orient='h')


# In[26]:


# Here how we will find the Data distribution
sns.distplot(data['price'])


# In[27]:


data['CarName'].unique()


# In[30]:


data['CarName']=data['CarName'].str.split('')


# In[32]:


data['CarName'].unique


# In[35]:


data['symboling'] # symoling gives u the maximum and minimum insurence which is on positive side which is safe and if is in negative which is not safe


# In[43]:


# cat and num columns names
cat_cols = data.select_dtypes(include=['object']).columns
num_cols = data.select_dtypes(exclude=['object']).columns
# cat and num data
cat_data = data[cat_cols]
num_data = data[num_cols]


# In[44]:


cat_data


# In[47]:


sns.pairplot(num_data)
# how the data is correlated with one another thats why we use pairplot


# In[48]:


data.columns


# In[49]:


data.head()


# In[50]:


sig_cat_col = ['fueltype', 'aspiration', 'carbody', 'drivewheel', 'cylindernumber']
dummies = pd.get_dummies(data[sig_cat_col])
dummies.shape


# In[52]:


dummies.head() # here we are doing one-hot-encoding


# In[53]:


dummies = pd.get_dummies(data[sig_cat_col], drop_first=True)
dummies.shape


# In[55]:


data.head() # here we have top drop the column car_ID


# In[56]:


del data['car_ID']


# In[58]:


data.head() # here see its not there car_ID 


# In[68]:


#del data['car_ID']
#del data['CarName']
data.drop(sig_cat_col, axis=1, inplace=True)


# In[67]:


data.head()


# In[69]:


data.enginelocation.unique()


# In[70]:


data.enginelocation.value_counts()


# In[71]:


data.doornumber.unique()


# In[72]:


data.doornumber.value_counts()


# In[79]:


#del data['car_ID']
#del data['CarName']
#del data['symboling']
#del data['doornumber']
#del data['enginelocation']
del data['fuelsystem']
data.drop(sig_cat_col, axis=1, inplace=True)


# In[80]:


data=pd.concat([data, dummies], axis=1)
data.head()


# In[83]:


data.info()


# # import the Data for Training and Testing

# In[91]:


# import all libraries and dependencies for machine learning
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.base import TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import r2_score


# In[92]:


# data split
df_train, df_test = train_test_split(data, train_size=0.7, random_state=42)


# In[93]:


df_train.head()


# In[94]:


df_train.price


# In[95]:


scaler = StandardScaler()


# In[97]:


sig_num_col = ['wheelbase', 'carlength', 'carwidth', 'curbweight', 'enginesize', 'boreratio', 'horsepower', 'citympg', 'highwaympg', 'price']
df_train[sig_num_col] = scaler.fit_transform(df_train[sig_num_col])


# In[98]:


df_train.head()


# In[101]:


# model building
y_train = df_train.pop('price')
X_train=df_train


# In[106]:


# Adding a constant variable and Build a first fitted model;
import statsmodels.api as sm
X_train_rfec = sm.add_constant(X_train)
lm_rfe = sm.OLS(y_train,X_train_rfec).fit()


#Summary of linear model
print(lm_rfe.summary())


# In[107]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif['Features'] =X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending=False)
vif


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




