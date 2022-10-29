#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[2]:


import os
os.chdir("C:/Users/user/Downloads")


# In[3]:


df=pd.read_csv('abalone.csv')


# In[4]:


df


# In[5]:


df.dtypes


# In[9]:


plt.scatter(df.index,df['Rings'])
plt.show()


# In[10]:


sns.boxplot(x='Length',y='Rings', data = df)
plt.show()


# In[11]:


sns.pairplot(df, hue="Length", height = 3)
plt.show()


# In[14]:


df = df[['Height','Rings']]
sns.heatmap(df.corr(), annot = True)
sns.set(rc = {'figure.figsize':(10,10)})


# In[16]:


df.isnull().sum()


# In[17]:


df.describe()


# In[18]:


df['Rings'].isnull().sum()


# In[20]:


df['Height'].mean()


# In[21]:


df['Height'].replace(np.NaN, df['Rings'].mean()).head(15)


# In[26]:


df['Rings'].median()


# In[23]:


df['Rings'].mode()


# In[24]:


from sklearn.preprocessing import OneHotEncoder


# In[25]:


encoder = OneHotEncoder(sparse=False)
cat_cols = ['Rings']


# In[27]:


from sklearn.preprocessing import StandardScaler
df_ready = df.copy()


# In[28]:


scaler = StandardScaler()
num_cols = ['Length','Diameter','Height','Whole weight','Shucked weight','viscera weight''Shell weight']


# In[29]:


df_ready.head()


# In[30]:


from sklearn.preprocessing import OneHotEncoder


# In[31]:


encoder = OneHotEncoder(sparse=False)
cat_cols = ['Rings']


# In[35]:


df_ready['Rings']=df_ready['Height'].apply(lambda x: 1 if x=='yes' else 0)


# In[36]:


print('shape of dataframe:',df_ready.shape)


# In[37]:


df_ready


# In[38]:


df = pd.get_dummies(df)
dummy_df = df


# In[39]:


from collections import Counter
def detection(df,features):
    outlier_indices = []
    
    for c in features:
        Q1 = np.percentile(df[c],26)
        
        Q3 = np.percentile(df[c],76)
        
        IQR = Q3 - Q1
        outlier_step = IQR * 1.5
        lower_range = Q1 - (outlier_step)
        upper_range = Q3 + (outlier_step)
        
        outlier_indices.extend(outlier_indices)
        
    outlier_indices = Counter(outlier_indicates)
    
    multiple_outliers = list(i for i, v in outlier_indices.times() if v > 2)
    
    return multiple_outliers


# In[40]:


df.info()


# In[41]:


x=df.iloc[:,:3]


# In[42]:


x


# In[43]:


df
y=df.iloc[:,1:]


# In[44]:


y


# In[45]:


feature = df_ready.drop('Rings', axis = 1)


# In[46]:


target = df_ready['Rings']


# In[47]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(feature, target, shuffle = True, test_size = 0.2, random_state = 1)


# In[48]:


print('Shape of training features:', X_train.shape)
print('Shape of training features:', X_test.shape)
print('Shape of training label:', y_train.shape)
print('Shape of training label:', y_test.shape)


# In[50]:


X_train


# In[51]:


y_train


# In[52]:


X_train.shape


# In[53]:


y_train.shape


# In[57]:


X_train = X_train.values.reshape((-1,1))


# In[58]:


X_train


# In[59]:


y_train


# In[60]:


X_test


# In[61]:


y_test


# In[62]:


transformed_Sex_feature = OneHotEncoder().fit_transform(df['Sex']).values.reshape((-1,1))
df_Sex_encoded = pd.DataFrame(transfprmed_Sex_features,coloumns = "Sex_"+str(int(i)) for i in random value 
df=  pd.concat([df_Sex_encoded], axis = 1)


# In[63]:


df.head()


# In[64]:


df['Diameter'] = df['Height'] + 1.5
df['Diameter'].head(5)


# In[ ]:




