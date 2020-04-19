#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


test_data = pd.read_csv("test.csv")


# In[3]:


test_data.head()


# In[4]:


train_data = pd.read_csv("train.csv")


# In[5]:


train_data.head()


# In[6]:


train_data.describe()


# In[7]:


train_data.info()


# In[8]:


test_data.describe()


# In[9]:


test_data.info()


# In[10]:


test_data.shape


# In[11]:


train_data.shape


# In[12]:


train_data.isnull().sum()


# In[13]:


test_data.isnull().sum()


# In[14]:


import matplotlib.pyplot as plt


# In[15]:


import seaborn as sns


# In[16]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[17]:


sns.set()


# In[18]:


def bar_chart(features):
    survived= train_data[train_data['Survived']==1][features].value_counts()
    death= train_data[train_data['Survived']==0][features].value_counts()
    df= pd.DataFrame([survived,death])
    df.plot(kind='bar' ,stacked=True ,figsize=(10,5))


# In[19]:


bar_chart('Sex')


# In[20]:


bar_chart('Parch')


# In[21]:


train_test_data =[train_data,test_data]
for dataset in train_test_data:
    dataset['Title'] = dataset['Name'].str.extract('([A-Za-z]+)\.',expand=False)


# In[22]:


train_data['Title'].value_counts()


# In[23]:


test_data['Title'].value_counts()


# mr :0
# miss:1
# mrs:2
# others:3

# In[24]:


title_mapping={"Mr":0,"Miss":1,"Mrs":2,"Master":3,"Dr":3,"Rev":3,"Mlle":3,"Col":3,"Major":3,
               "Ms":3,"Capt":3,"Sir":3,"Countess":3,"Lady":3,"Don":3,"Mme":3,"Jonkheer":3,"Dona":3}
for dataset in train_test_data:
    dataset['Title']=dataset["Title"].map(title_mapping)


# In[25]:


train_data.head()


# In[26]:


test_data.head()


# In[27]:


test_data.head()


# In[28]:


bar_chart('Title')


# In[29]:


train_data.drop('Name', axis=1, inplace=True)
test_data.drop('Name', axis=1, inplace=True)


# In[30]:


train_data.head()


# In[31]:


test_data.head()


# In[32]:


sex_mapping ={"male":0,"female":1}
for dataset in train_test_data:
    dataset["Sex"]= dataset["Sex"].map(sex_mapping)


# In[33]:


bar_chart('Sex')


# In[34]:


train_data.head()


# In[35]:


train_data['Age'].fillna(train_data.groupby('Title')["Age"].transform("median"),inplace=True)
test_data['Age'].fillna(test_data.groupby('Title')["Age"].transform("median"),inplace=True)


# In[36]:


train_data.isnull().sum()


# In[37]:


test_data.isnull().sum()


# child:0
# young:1
# adult:2
# senoir:3
# aged:4

# In[38]:


for dataset in train_test_data:
    dataset.loc[ dataset ['Age']<=18 ,"Age"]=0,
    dataset.loc[( dataset ['Age']>18) & (dataset['Age']<=30) ,"Age"]=1,
    dataset.loc[( dataset ['Age']>30) & (dataset['Age']<=40) ,"Age"]=2,
    dataset.loc[( dataset ['Age']>40) & (dataset['Age']<=60) ,"Age"]=3,
    dataset.loc[dataset["Age"]>60,"Age"]=4
   


# In[39]:


train_data.head()


# In[40]:


test_data.head()


# In[41]:


bar_chart("Pclass")


# In[42]:


Pclass1= train_data[train_data['Pclass']==1]["Embarked"].value_counts()
Pclass2= train_data[train_data['Pclass']==2]["Embarked"].value_counts()
Pclass3= train_data[train_data['Pclass']==3]["Embarked"].value_counts()
df=pd.DataFrame([Pclass1,Pclass2,Pclass3])
df.index=['1st Class','2nd Class','3rd Class']
df.plot(kind='bar',stacked=True,figsize=(10,5))


# In[43]:


for dataset in train_test_data:
    dataset["Embarked"]=dataset['Embarked'].fillna("S")


# In[44]:


test_data['Fare'].fillna(test_data.groupby('Cabin')['Fare'].transform("median"),inplace=True)


# In[45]:


train_data.isnull().sum()


# In[46]:


test_data.isnull().sum()


# In[47]:


train_data.head()


# In[48]:


for dataset in train_test_data:
    dataset.loc[dataset['Fare']<=17 ,"Fare"]=0,
    dataset.loc[(dataset["Fare"]>=17 ) & (dataset["Fare"]<30),"Fare"]=1,
    dataset.loc[(dataset["Fare"]>=30 ) & (dataset["Fare"]<100),"Fare"]=2,
    dataset.loc[(dataset["Fare"]>=100),"Fare"]=3,


# In[49]:


train_data.head()


# In[50]:


train_data.Cabin.value_counts()


# In[51]:


Pclass1= train_data[train_data['Pclass']==1]["Cabin"].value_counts()
Pclass2= train_data[train_data['Pclass']==2]["Cabin"].value_counts()
Pclass3= train_data[train_data['Pclass']==3]["Cabin"].value_counts()
df=pd.DataFrame([Pclass1,Pclass2,Pclass3])
df.index=['1st Class','2nd Class','3rd Class']
df.plot(kind='bar',stacked=True,figsize=(10,5))


# In[52]:


cabin_mapping={"A":0,"B":0.4,"c":0.8,"D":1.2,"E":1.6,"F":2,"G":2.4,"T":2.8}
for dataset in train_test_data:
    dataset["Cabin"]=dataset["Cabin"].map(cabin_mapping)


# In[53]:


train_data.drop('Ticket', axis=1, inplace=True)


# In[54]:


test_data.drop('Ticket', axis=1, inplace=True)


# In[55]:


train_data.drop("PassengerId", axis=1,inplace=True)


# In[56]:


test_data.drop("Cabin" ,axis=1,inplace=True)
train_data.drop("Cabin",axis=1,inplace=True)
test_data['Fare'].fillna(test_data.groupby('Pclass')['Fare'].transform("median"),inplace=True)


# In[57]:


test_data.isnull().sum()


# In[58]:


train_data.isnull().sum()


# In[59]:


train_data.head()


# In[60]:


embarked_mapping={"S":0,"C":1,"Q":2}
for dataset in train_test_data:
    dataset['Embarked']=dataset["Embarked"].map(embarked_mapping)


# In[61]:


X_train = train_data.drop('Survived', axis=1)
y_train = train_data['Survived']
X_test = test_data.drop('PassengerId', axis=1).copy()


# In[73]:


from sklearn.svm import SVC,LinearSVC


# In[74]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np


# In[63]:


train_data.info()


# In[64]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
k_fold=KFold(n_splits=10, shuffle=True,random_state=0)
from sklearn.linear_model import LogisticRegression


# In[72]:


clf=LogisticRegression()
clf.fit(X_train,y_train)
y_pred_logress = clf.predict(X_test)
acc_log= round(clf.score(X_train,y_train)*100,2)
print(str(acc_log)+'percentage')


# In[75]:


clf=SVC()
clf.fit(X_train,y_train)
y_pred_svc=clf.predict(X_test)
acc_svc=round(clf.score(X_train,y_train)*100,2)
print(str(acc_svc)+'percentage')


# In[77]:


clf=LinearSVC()
clf.fit(X_train,y_train)
y_pred_linear_svc=clf.predict(X_test)
acc_linear_svc=round(clf.score(X_train,y_train)*100,2)
print(str(acc_linear_svc)+'percentage')


# In[78]:


clf=KNeighborsClassifier()
clf.fit(X_train,y_train)
y_pred_knn=clf.predict(X_test)
acc_knn=round(clf.score(X_train,y_train)*100,2)
print(str(acc_knn)+'percentage')


# In[79]:


clf=DecisionTreeClassifier()
clf.fit(X_train,y_train)
y_pred_decision_tree=clf.predict(X_test)
acc_decision_tree=round(clf.score(X_train,y_train)*100,2)
print(str(acc_decision_tree)+'percentage')


# In[87]:


clf=RandomForestClassifier()
clf.fit(X_train,y_train)
y_pred_forest=clf.predict(X_test)
acc_forest=round(clf.score(X_train,y_train)*100,2)
print(str(acc_forest)+'percentage')


# In[89]:


models= pd.DataFrame({"Model":["RandomForestClassifier","DecisionTreeClassifier",
                               "KNeighborsClassifier","LinearSVC","SVC","LogisticRegression"],
                     "Score":[acc_forest,acc_decision_tree,acc_knn,acc_linear_svc,acc_svc,acc_log]})
models.sort_values(by='Score',ascending=False)


# In[91]:


test_data.head()


# In[93]:


submission=pd.DataFrame({"passengerId":test_data["PassengerId"],"Survived":y_pred_forest})


# In[96]:


submission.to_csv('submission.csv',index=False)


# In[ ]:




