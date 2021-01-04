#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd


# In[3]:


df1=pd.read_csv('train.csv')


# In[4]:


df1.head()


# In[5]:


df1.drop(['PassengerId','Name','Ticket','Cabin','Embarked','SibSp','Parch'],axis=1,inplace=True)


# In[6]:


df1.head()


# In[7]:


df1.isnull().sum()


# In[8]:


df1['Age'].describe()


# In[9]:


df1['Age'].fillna(df1['Age'].mean(),inplace=True)


# In[10]:


df1.isnull().sum()


# In[11]:


sex_dummies=pd.get_dummies(df1['Sex'],drop_first=True)


# In[12]:


df1=pd.concat([df1,sex_dummies],axis=1)


# In[13]:


df1.head()


# In[15]:


df1.drop(['Sex'],axis=1,inplace=True)


# In[16]:


df1.head()


# In[18]:


from sklearn.preprocessing import StandardScaler
sts =StandardScaler()


# In[19]:


feature_scale=['Age','Fare']
df1[feature_scale] = sts.fit_transform(df1[feature_scale])


# In[20]:


df1.head()


# In[21]:


X=df1.drop(['Survived'],axis=1)
y=df1['Survived']


# In[22]:


from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


# In[23]:


#create param
model_param = {
    'DecisionTreeClassifier':{
        'model':DecisionTreeClassifier(),
        'param':{
            'criterion': ['gini','entropy']
        }
    },
        'KNeighborsClassifier':{
        'model':KNeighborsClassifier(),
        'param':{
            'n_neighbors': [5,10,15,20,25]
        }
    },
        'SVC':{
        'model':SVC(),
        'param':{
            'kernel':['rbf','linear','sigmoid'],
            'C': [0.1, 1, 10, 100]
         
        }
    }
}


# In[24]:


scores =[]
for model_name, mp in model_param.items():
    model_selection = GridSearchCV(estimator=mp['model'],param_grid=mp['param'],cv=5,return_train_score=False)
    model_selection.fit(X,y)
    scores.append({
        'model': model_name,
        'best_score': model_selection.best_score_,
        'best_params': model_selection.best_params_
    })


# In[25]:


df_model_score = pd.DataFrame(scores,columns=['model','best_score','best_params'])
df_model_score


# In[26]:


model_svc = SVC( C= 100,kernel='rbf')


# In[27]:


model_svc.fit(X, y)


# In[29]:


df2 = pd.read_csv('test.csv')


# In[30]:


df2.head()


# In[31]:


df3=df2.drop(['PassengerId','Name','Ticket','Cabin','Embarked','SibSp','Parch'], axis=1 )


# In[32]:


df3.isnull().sum()


# In[34]:


df3['Age'].fillna(df3['Age'].mean(),inplace=True)
df3['Fare'].fillna(df3['Fare'].mean(),inplace=True)


# In[35]:


l_sex_dummies=pd.get_dummies(df3['Sex'],drop_first=True)
df3= pd.concat([df3,l_sex_dummies],axis=1)
df3.drop(['Sex'], axis=1, inplace=True )


# In[36]:


df3.head()


# In[37]:


df3[feature_scale] = sts.fit_transform(df3[feature_scale])


# In[38]:


df3.head()


# In[39]:


y_predicted = model_svc.predict(df3)


# In[40]:


submission = pd.DataFrame({
        "PassengerId": df2['PassengerId'],
        "Survived": y_predicted
    })


# In[42]:


submission.to_csv('titanic_sub.csv', index=False)


# In[ ]:




