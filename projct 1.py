#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


data = pd.read_csv('winequality-red.csv')

data.head()


# In[5]:




data['quality'].value_counts()


# In[8]:


import seaborn as sns


# In[9]:


f, ax = plt.subplots(figsize=(10, 8))
corr = data.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)


# In[10]:


sns.pairplot(data)


# In[11]:


data.columns


# In[12]:


data['quality'] = data['quality'].map({3 : 'bad', 4 :'bad', 5: 'bad',
                                      6: 'good', 7: 'good', 8: 'good'})


# In[13]:


data['quality'].value_counts()


# In[14]:


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

data['quality'] = le.fit_transform(data['quality'])

data['quality'].value_counts


# In[15]:


sns.countplot(data['quality'])


# In[16]:


x = data.iloc[:,:11]
y = data.iloc[:,11]

# determining the shape of x and y.
print(x.shape)
print(y.shape)


# In[20]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 44)

# determining the shapes of training and testing sets
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# In[22]:


from sklearn.ensemble import RandomForestClassifier


# In[23]:


model = RandomForestClassifier(n_estimators = 200)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print("Training accuracy :", model.score(x_train, y_train))
print("Testing accuracy :", model.score(x_test, y_test))


# In[24]:


from sklearn.metrics import classification_report, confusion_matrix

from sklearn.model_selection import GridSearchCV, cross_val_score


# In[25]:


print(classification_report(y_test, y_pred))


print(confusion_matrix(y_test, y_pred))


# In[26]:


model_eval = cross_val_score(estimator = model, X = x_train, y = y_train, cv = 10)
model_eval.mean()


# In[ ]:




