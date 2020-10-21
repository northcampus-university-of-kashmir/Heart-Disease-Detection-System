#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# 
# # Loading the data

# In[2]:


df = pd.read_csv("heart.csv")


# In[3]:


df.head()


# #  Exploratory Data Analysis (EDA)

# In[4]:


info = ["age","1: male, 0: female","chest pain type, 1: typical angina, 2: atypical angina, 3: non-anginal pain,4: asymptomatic",
"resting blood pressure"," serum cholestoral in mg/dl","fasting blood sugar > 120 mg/dl","resting electrocardiographic results (values 0,1,2)",
" maximum heart rate achieved","exercise induced angina","oldpeak = ST depression induced by exercise relative to rest",
"the slope of the peak exercise ST segment","number of major vessels (0-3) colored by flourosopy",
"thal: 3 = normal; 6 = fixed defect; 7 = reversable defect"]



for i in range(len(info)):
    print(df.columns[i]+":\t\t\t"+info[i])


# In[5]:


df.describe()


# In[6]:


df.info()


# There are no null values in our dataset

# In[7]:


df.nunique()


# In[8]:


df.shape


# In[9]:


df.target.value_counts()


# # Data Visulization

# In[10]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[11]:


plt.figure(figsize=(5,5))

df.target.value_counts().plot(kind="bar", color=["r", "b"],edgecolor="y",linewidth=2,hatch='/')
plt.ylim(0,200)
plt.title(" Target feature (total count of  having heart disease or not)",fontweight="bold",color="g")
plt.xlabel(" \nHeart disease = 1 , Not heart disease = 0 ",fontweight="bold",color="y")
plt.ylabel(" count of person having disease or not  ",fontweight="bold")

print("\nWe have 165 person with Heart disease and 138 person without heart disease, so our dataset is balanced")


# In[12]:


plt.title("total number of male and female",fontweight="bold")
plt.ylabel("total number of male and female ",fontweight="bold",color="m")
plt.xlabel("male=1   female=0",fontweight="bold",color="r")
df["sex"].value_counts().plot(kind="bar",color=["g","c"],edgecolor="y",linewidth=2,hatch='/');


# In[13]:



plt.figure(figsize=(12,12))
plt.subplot(2,2,1)
plt.title("no of males  having / not having heart disease",fontweight="bold",color="c")
plt.ylabel("total number of male  ",fontweight="bold",color="m")
plt.xlabel("0= no disease ,  1= disease in males ",fontweight="bold",color="b")
plt.ylim(60,140)
df[df["sex"] == 1]["target"].value_counts().plot(kind="bar",color=["b","r"],edgecolor="y",linewidth=2,hatch='/')

#plt.legend()
plt.subplot(2,2,2)
plt.title("no of females having / not having heart disease",fontweight="bold",color="c")
plt.ylabel("total number of female  ",fontweight="bold",color="m")
plt.xlabel("1=  disease ,  0= no disease in females ",fontweight="bold",color="b")

df[df["sex"] == 0]["target"].value_counts().plot(kind="bar",color=["r","b"],edgecolor="y",linewidth=2,hatch='/')
plt.show()
#df[df["sex"] == 1]["target"].value_counts().plot(kind="bar",color="r")
#plt.savefig("pplhaving.png")


# In[88]:


df.hist(figsize=(15,15),color="g",edgecolor="y",linewidth=2,linestyle='-');


# In[89]:


plt.figure(figsize=(15,6))
sns.countplot(x='age',data = df, hue = 'target',palette='cubehelix')
plt.show()


# In[16]:


categorical_val = []
continous_val = []
for column in df.columns:
    print('==============================')
    print(f"{column} : {df[column].unique()}")
    if len(df[column].unique()) <= 10:
        categorical_val.append(column)
    else:
        continous_val.append(column)


# In[17]:


print(" categorical val ",categorical_val)
print("continous val ",continous_val)


# # Ploting categorical features 

# In[91]:


plt.figure(figsize=(15, 15))
for i, column in enumerate(categorical_val,1):
    plt.subplot(3, 3, i)
    df[df["target"] == 0][column].hist(bins=35, color='blue', label='Have Heart Disease = NO', alpha=0.6)
    df[df["target"] == 1][column].hist(bins=35, color='red', label='Have Heart Disease = YES', alpha=0.6)
    plt.legend()
    plt.xlabel(column)


# 1.cp {Chest Pain} : People with cp equl to 1, 2, 3 are more likely to have heart disease than people with cp equal to 0.
# 
# 2.restecg {resting electrocardiographic results} : People with value 1 (signals non-normal heart beat, can range from mild symptoms to severe problems) are more likely to have heart disease.
# 
# 3.exang {exercise induced angina} : People with value 0 (No ==> exercice induced angina) have heart disease more than people with value 1 (Yes ==> exercice induced angina)
# 
# 4.slope {the slope of the peak exercise ST segment} : People with slope value equal to 2 (Downslopins: signs of unhealthy heart) are more likely to have heart disease than people with slope value equal to 0 (Upsloping: better heart rate with excercise) or 1 (Flatsloping: minimal change (typical healthy heart)).
# 
# 5.ca {number of major vessels (0-3) colored by flourosopy} : the more blood movement the better so people with ca equal to 0 are more likely to have heart disease.
# 
# 6.thal {thalium stress result} : People with thal value equal to 2 (fixed defect: used to be defect but ok now) are more likely to have heart disease.

# # ploting continous features below

# In[19]:


plt.figure(figsize=(15, 15))

for i, column in enumerate(continous_val, 1):
    plt.subplot(3, 2, i)
    df[df["target"] == 0][column].hist(bins=35, color='blue', label='Have Heart Disease = NO', alpha=0.6)
    df[df["target"] == 1][column].hist(bins=35, color='red', label='Have Heart Disease = YES', alpha=0.6)
    plt.legend()
    plt.xlabel(column)


# 1.trestbps : resting blood pressure (in mm Hg on admission to the hospital) anything above 130-140 is typically cause for concern
# 
# 2.chol {serum cholestoral in mg/dl} : above 200 is cause for concern.
# 
# 3.thalach {maximum heart rate achieved} : People that have acheived a maximum more than 140 are more likely to have heart disease.
# 
# 4.oldpeak ST depression induced by exercise relative to rest looks at stress of heart during excercise unhealthy heart will stress more

# # Age vs. Max Heart Rate for Heart Disease

# In[20]:


# Create another figure
plt.figure(figsize=(10, 8))

# Scatter with postivie examples
plt.scatter(df.age[df.target==1],
            df.thalach[df.target==1],
            c="r")

# Scatter with negative examples
plt.scatter(df.age[df.target==0],
            df.thalach[df.target==0],
            c="blue")

# Add some helpful info
plt.title("Heart Disease in function of Age and Max Heart Rate",fontweight="bold")
plt.xlabel("Age")
plt.ylabel("Max Heart Rate")
plt.legend(["Disease", "No Disease"]);
#terminator elimnates <matplotlib.legend.Legend at 0x7f39518951d0>


# # Correlation Matrix (heat map)

# In[21]:


corr_matrix = df.corr()
fig, ax = plt.subplots(figsize=(12, 12))
ax = sns.heatmap(corr_matrix,
                 annot=True,
                 linewidths=0.5,
                 fmt=".2f",
                 cmap="YlGnBu");
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.3, top - 0.3)


# In[22]:


df.drop('target', axis=1).corrwith(df.target).plot(kind='bar', grid=True, figsize=(12, 8), 
                                                   title="Correlation with target",color="g");


# 
# 1.fbs and chol are the lowest correlated with the target variable.
# 
# 2.All other variables have a significant correlation with the target variable.
# 

# # Data processing

# # One hot encoding

# In[23]:


if 'target'  in categorical_val:
    categorical_val.remove('target')
    dataset = pd.get_dummies(df, columns = categorical_val)
else:
    pass

dataset.head()


# In[24]:


print(df.columns)
print(dataset.columns)


# # Standardization

# In[25]:


from sklearn.preprocessing import StandardScaler

s_sc = StandardScaler()
col_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
dataset[col_to_scale] = s_sc.fit_transform(dataset[col_to_scale])


# # Performance of an Algorithm

# In[26]:


from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
train_score_1x={}

f1_train1x={}
precision_train1x={}
recall_train1x={}
test_score_1x={}
f1_test1x={}
precision_test1x={}
recall_test1x={}
def print_score(algo_name,clf, X_train, y_train, X_test, y_test, train=True):
    
    if train:
        pred = clf.predict(X_train)
        print("Train Result:\n================================================")
        
        print(f"Accuracy Score: {accuracy_score(y_train, pred) * 100:.2f}%")
        print("_______________________________________________")
        print("Classification Report:", end='')
        print(f"\tPrecision Score: {precision_score(y_train, pred) * 100:.2f}%")
        print(f"\t\t\tRecall Score: {recall_score(y_train, pred) * 100:.2f}%")
        print(f"\t\t\tF1 score: {f1_score(y_train, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_train, pred)}\n")
        train_score_1x["Training score "+algo_name]=accuracy_score(y_train, pred) * 100
        f1_train1x[algo_name+" training f1 score "]=f1_score(y_train, pred) * 100
        precision_train1x[algo_name+" training precision score"]=precision_score(y_train, pred) * 100
        recall_train1x[algo_name+" training recall score"]= recall_score(y_train, pred) * 100
        
    elif train==False:
        pred = clf.predict(X_test)
        print("Test Result:\n================================================")        
        print(f"Accuracy Score: {accuracy_score(y_test, pred) * 100:.2f}%")
        print("_______________________________________________")
        print("Classification Report:", end='')
        print(f"\tPrecision Score: {precision_score(y_test, pred) * 100:.2f}%")
        print(f"\t\t\tRecall Score: {recall_score(y_test, pred) * 100:.2f}%")
        print(f"\t\t\tF1 score: {f1_score(y_test, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_test, pred)}\n")
        test_score_1x["Testing score"+algo_name]=accuracy_score(y_test, pred) * 100
        f1_test1x[algo_name+" testing f1 score"]=f1_score(y_test, pred) * 100
        precision_test1x[algo_name+" testing precision score"]=precision_score(y_test, pred) * 100
        recall_test1x[algo_name+" testing recall score"]= recall_score(y_test, pred) * 100
    #return f1_train , precision_train ,recall_train,f1_test,precision_test,recall_test


# # Splitting data into train and test set

# In[27]:


from sklearn.model_selection import train_test_split

X = dataset.drop('target', axis=1)
y = dataset.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[28]:


X_test


# Now we've got our data split into training and test sets, it's time to build a machine learning model.
# 
# We'll train it (find the patterns) on the training set.
# 
# And we'll test it (use the patterns) on the test set.
# 
# We're going to try  different machine learning models:
# 
# Logistic Regression
# K-Nearest Neighbours Classifier
# Decision Tree Classifier
# Random Forest Classifier
# XGBoost Classifier

# # Applying  Algorithms (without hypertuning)

# # Decsion tree

# In[29]:


from sklearn.tree import DecisionTreeClassifier


tree2 = DecisionTreeClassifier(random_state=42)
tree2.fit(X_train, y_train)

print_score("*dt ",tree2, X_train, y_train, X_test, y_test, train=True)
print_score("*dt ",tree2, X_train, y_train, X_test, y_test, train=False)


# # Random Forest below

# In[30]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

rand_forest2 = RandomForestClassifier(n_estimators=1000, random_state=42)
rand_forest2.fit(X_train, y_train)

print_score("*random\nforest",rand_forest2, X_train, y_train, X_test, y_test, train=True)
print_score("*random\n forest",rand_forest2, X_train, y_train, X_test, y_test, train=False)


# # KNN

# In[31]:


from sklearn.neighbors import KNeighborsClassifier

knn_classifier2 = KNeighborsClassifier()
knn_classifier2.fit(X_train, y_train)

print_score("*KNN",knn_classifier2, X_train, y_train, X_test, y_test, train=True)
print_score("*KNN",knn_classifier2, X_train, y_train, X_test, y_test, train=False)


# # Logistic regression

# In[32]:


from sklearn.linear_model import LogisticRegression

log_reg2 = LogisticRegression(solver='liblinear')
log_reg2.fit(X_train, y_train);
print_score("*log-reg",log_reg2, X_train, y_train, X_test, y_test, train=True)
print_score("*log-reg",log_reg2, X_train, y_train, X_test, y_test, train=False)


# # SVM

# In[33]:


from sklearn.svm import SVC
svm2 = SVC(kernel='rbf', gamma=0.1, C=1.0)
svm2.fit(X_train, y_train);
print_score("*SVM",svm2, X_train, y_train, X_test, y_test, train=True)
print_score("*SVM",svm2, X_train, y_train, X_test, y_test, train=False)


# # xgboost

# In[34]:


from xgboost import XGBClassifier 
xgboost2 = XGBClassifier()
#xgboost1=RandomizedSearchCV(xgboost1,param_distributions=params,n_iter=5,scoring='roc_auc',n_jobs=-1,cv=5,verbose=3,iid=True);
#xgboost1.fit(X_train, y_train)


# In[35]:


xgboost2=XGBClassifier()
xgboost2.fit(X_train, y_train);


# In[36]:


print_score("*xgboost",xgboost2 , X_train, y_train, X_test, y_test, train=True)
print_score("*xgboost",xgboost2 , X_train, y_train, X_test, y_test, train=False)


# # Voting classifier

# In[37]:


from sklearn.ensemble import VotingClassifier 


# In[38]:


voting_clf2 = VotingClassifier( estimators=[
                                          ('lr',log_reg2 ), 
                                         ('knn',knn_classifier2 ),
                                          ('dt',tree2),
                                         ('rf',rand_forest2),
                                          ('xgb',xgboost2),
                                          
                                          ],
                                            voting ='soft')
voting_clf2.fit(X_train,y_train);


# In[39]:


for clf in (
            log_reg2,
            knn_classifier2,
            tree2,
            rand_forest2,
            xgboost2,
            voting_clf2):
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred)*100)


# In[40]:


print_score("*voting",voting_clf2, X_train, y_train, X_test, y_test, train=True)
print_score("*voting",voting_clf2, X_train, y_train, X_test, y_test, train=False)


# # Algorithm performance before hypertuning

# So as we have dictionries in print_score()  function 
# we will extract value from these dictionaries for 
# ploting purpose  
# those dictionaries are
# train_score_1x={}
# f1_train1x={}
# precision_train1x={}
# recall_train1x={}
# test_score_1x={}
# f1_test1x={}
# precision_test1x={}
# recall_test1x={}

# In[41]:


#test_scores=[test_score1,test_score2,test_score3,test_score4,test_score5]


algos_names1=["Decision\ntree","Random\nforest", "KNN","Logistic\nregression","SVM","xgboost"]
train_score_1=list(train_score_1x.values())
test_score_1=list(test_score_1x.values())
f1_test1=list(f1_test1x.values())
precision_test1=list(precision_test1x.values())
recall_test1=list(recall_test1x.values())
f1_train1=list(f1_train1x.values())
precision_train1=list(precision_train1x.values())
recall_train1=list(recall_train1x.values())


# In[42]:


test_score_1


# Genrating csv of above algorithms

# In[43]:


results_df = pd.DataFrame(data=[["Random forest",test_score_1[1],f1_test1[1],precision_test1[1],recall_test1[1]],
                                ["KNN",test_score_1[2],f1_test1[2],precision_test1[2],recall_test1[2]],
                                ["Decision tree",test_score_1[0],f1_test1[0],precision_test1[0],recall_test1[0]],
                                ["Logistic Regression", test_score_1[3],f1_test1[3],precision_test1[3],recall_test1[3]],
                                ["SVM", test_score_1[4],f1_test1[4],precision_test1[4],recall_test1[4]] ,
                                ["Xgboost", test_score_1[5],f1_test1[5],precision_test1[5],recall_test1[5]] ,
                                ["Voting", test_score_1[6],f1_test1[6],precision_test1[6],recall_test1[6]] 
                                
                               ], 
                          columns=['Model', 'Testing Accuracy %'
                                  ,'f1 test %','Precision test %','Recall test %'])



#results_df.to_csv("results/*trainresults.csv") 

results_df


# # Algorithms  with HYPERPARAMETER TUNING 

# # Decision tree 

# In[44]:


from sklearn. tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV ,RandomizedSearchCV

params = {"criterion":("gini", "entropy"), 
          "splitter":("best", "random"), 
          "max_depth":(list(range(1, 20))), 
          "min_samples_split":[2, 3, 4], 
          "min_samples_leaf":list(range(1, 20))
          }


tree1 = DecisionTreeClassifier(random_state=42)
grid_search_cv = GridSearchCV(tree1, params, scoring="accuracy", verbose=3, n_jobs=-1, cv=3,iid=True)


# Verbose is a general programming term for produce lots of logging output.
# You can think of it as asking the program to "tell me everything about what you are doing all the time".
# Just set it to true and see what happens.
# 
# iid : boolean, default=True
# 
#     If True, the data is assumed to be identically distributed across the folds, and the loss minimized is the total 
#     loss per sample, and not the mean loss across the folds.
# 
# n_jobs : int, default=1
# 
#     Number of jobs to run in parallel.

# In[45]:


#grid_search_cv.fit(X_train, y_train)


# In[46]:


#grid_search_cv.best_estimator_ 


# In[47]:


#grid_search_cv.best_score_
#grid_search_cv.best_params_


# 
# best_estimator_ : estimator -> Estimator that was chosen by the search, 
#                 i.e. estimator which gave highest score 
#                 (or smallest loss if specified) on the left 
#                 out data. Not available if refit=False.
# 
# best_score_ : float
# 
#     Score of best_estimator on the left out data.
# 
# best_params_ : dict
# 
#     Parameter setting that gave the best results on the hold out data.
# 
# scorer_ : function
# 
#     Scorer function used on the held out data to 
#     choose the best parameters for the model

# In[48]:


tree1 = DecisionTreeClassifier(criterion='gini', 
                              max_depth=3,
                              min_samples_leaf=2, 
                              min_samples_split=2, 
                              splitter='random')


# In[49]:


tree1.fit(X_train, y_train)


print_score("Decision tree",tree1,X_train, y_train, X_test, y_test, train=True)
print_score("Decision tree",tree1, X_train, y_train, X_test, y_test, train=False)


# In[ ]:





# # logistic regression

# In[50]:


from sklearn.linear_model import LogisticRegression
"""
params1 = {"solver":("newton-cg", "lbfgs", "liblinear", "sag", "saga"), 
           
          "C":(list(range(1, 100)))  ,
           "multi_class":("auto", "ovr")
          }
"""
log_reg = LogisticRegression(random_state=42,solver="liblinear")

#grid_search_cv = GridSearchCV(log_reg, params1, scoring="accuracy", verbose=3, n_jobs=-1, cv=3,iid=True)

#grid_search_cv.fit(X_train, y_train)


log_reg.fit(X_train,y_train)


# In[51]:


#grid_search_cv.best_estimator_ 


# In[52]:


log_reg = LogisticRegression(C=1,solver="newton-cg")
log_reg.fit(X_train,y_train)


# In[53]:


print_score("Logistic regression",log_reg, X_train, y_train, X_test, y_test, train=True)
print_score("Logistic regression",log_reg, X_train, y_train, X_test, y_test, train=False)


# # Svm

# In[54]:


from sklearn import svm


# In[55]:


params= {'C': [0.1, 1, 10, 100, 1000],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf']}  


# In[56]:


svm1 = svm.SVC(kernel='linear') 
#svmrn=RandomizedSearchCV(svm1,param_distributions=params,n_iter=5,scoring='roc_auc',n_jobs=-1,cv=5,verbose=3,iid=True);


# In[57]:


#svmrn.fit(X_train, y_train)


# In[58]:


#svmrn.best_estimator_


# In[59]:


svm1=svm.SVC(C=10, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma=0.001, kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)


# In[60]:


svm1.fit(X_train, y_train)


# In[61]:


print_score("SVM",svm1, X_train, y_train, X_test, y_test, train=True)
print_score("SVM",svm1, X_train, y_train, X_test, y_test, train=False)


# # KNN algorithm

# In[62]:


from sklearn.neighbors import KNeighborsClassifier

train_score = []
test_score = []
neighbors = range(1, 21)

for k in neighbors:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    train_score.append(accuracy_score(y_train, model.predict(X_train)))
    test_score.append(accuracy_score(y_test, model.predict(X_test)))


# In[63]:


plt.figure(figsize=(5,5))

plt.plot(neighbors, train_score, label="Train score")
plt.plot(neighbors, test_score, label="Test score")
plt.xticks(np.arange(1, 21, 1))
plt.xlabel("Number of neighbors")
plt.ylabel("Model score")
plt.legend()

print(f"Maximum KNN score on the test data: {max(test_score)*100:.2f}%")


# In[64]:


knn_classifier = KNeighborsClassifier(n_neighbors=19)
knn_classifier.fit(X_train, y_train)

print_score("KNN",knn_classifier, X_train, y_train, X_test, y_test, train=True)
print_score("KNN",knn_classifier, X_train, y_train, X_test, y_test, train=False)


# # Random forest

# In[65]:


from sklearn.ensemble import RandomForestClassifier

"""
n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]

random_grid = {'n_estimators': n_estimators, 'max_features': max_features,
               'max_depth': max_depth, 'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf, 'bootstrap': bootstrap}

rand_forest = RandomForestClassifier()

grid_search_cv2 = GridSearchCV(rand_forest, random_grid, scoring="accuracy", verbose=3, n_jobs=-1, cv=3,iid=True)

"""


#  n_estimators = number of trees in the foreset
#     
#     max_features = max number of features considered for splitting a node
#     
#     max_depth = max number of levels in each decision tree
#     
#     min_samples_split = min number of data points placed in a node before the node is split
#     
#     min_samples_leaf = min number of data points allowed in a leaf node
#     
#     bootstrap = method for sampling data points (with or without replacement)

# In[66]:


#grid_search_cv2.fit(X_train, y_train)


# In[67]:


#grid_search_cv2.best_params_
#rand_forest.fit(X_train, y_train)

rand_forest = RandomForestClassifier(bootstrap= True,max_depth= 10, min_samples_split= 5, n_estimators= 200)


# In[68]:


rand_forest.fit(X_train,y_train);


# In[69]:


print_score("Random forest",rand_forest, X_train, y_train, X_test, y_test, train=True)
print_score("Random forest",rand_forest, X_train, y_train, X_test, y_test, train=False)


# In[ ]:





# In[ ]:





# # Xgboost

# In[70]:


params={
    "learning_rate" :  [0.05,0.10,0.15,0.20,0.25,0.30],
    "max_depth" : [3,4,5,6,8,10,12,15],
    "min_child_weight" : [1,3,5,7],
    "gamma" : [0.0,0.1,0.2,0.3,0.4],
    "colsample_bytree" : [0.3,0.4,0.5,0.7]
}


# In[ ]:





# In[71]:


from xgboost import XGBClassifier 
xgboost1 = XGBClassifier()
#xgboost1=RandomizedSearchCV(xgboost1,param_distributions=params,n_iter=5,scoring='roc_auc',n_jobs=-1,cv=5,verbose=3,iid=True);
#xgboost1.fit(X_train, y_train)


# In[72]:


#xgboost1.best_estimator_


# In[73]:


xgboost1=XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.7, gamma=0.4,
              learning_rate=0.25, max_delta_step=0, max_depth=8,
              min_child_weight=7, missing=None, n_estimators=100, n_jobs=1,
              nthread=None, objective='binary:logistic', random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
              silent=None, subsample=1, verbosity=1)
xgboost1.fit(X_train, y_train);


# In[ ]:





# In[74]:


print_score("xgboost",xgboost1, X_train, y_train, X_test, y_test, train=True)
print_score("xgboost",xgboost1, X_train, y_train, X_test, y_test, train=False)


# In[ ]:





# In[ ]:





# In[ ]:





# # Voting classifier

# In[75]:


from sklearn.ensemble import VotingClassifier 


# In[76]:


voting_clf = VotingClassifier( estimators=[
                                          ('lr',log_reg ), 
                                         ('knn',knn_classifier ),
                                          ('dt',tree1),
                                         ('rf',rand_forest),
                                          ('xgb',xgboost1),
                                          
                                          ],
                                            voting ='soft')
voting_clf.fit(X_train,y_train);


# In[77]:


for clf in (
            log_reg,
            knn_classifier,
            tree1,
            rand_forest,
            xgboost1,
            voting_clf):
    voting_clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred)*100)


# In[ ]:





# In[ ]:





# In[78]:


print_score("Voting Classifier", voting_clf,X_train, y_train, X_test, y_test, train=True)
print_score("Voting Classifier",voting_clf, X_train, y_train, X_test, y_test, train=False)


# # Comparing effect of hyperparameter tuning on our model

# below are returned scores from print_scores() fucntion

# In[ ]:





# In[79]:


#test_scores=[test_score1,test_score2,test_score3,test_score4,test_score5]


algos_names=["Dt", "Log-reg","SVM", "KNN", "Rnf","xgboost", "Voting"]
train_score_=list(train_score_1x.values())[7:]

test_score_=list(test_score_1x.values())[7:]
f1_test=list(f1_test1x.values())[7:]
precision_test=list(precision_test1x.values())[7:]
recall_test=list(recall_test1x.values())[7:]
f1_train=list(f1_train1x.values())[7:]
precision_train=list(precision_train1x.values())[7:]
recall_train=list(recall_train1x.values())[7:]


# In[80]:


test_score_1x


# In[ ]:





# In[81]:


results_df1 = pd.DataFrame(data=[["Random forest",test_score_[4],f1_test[4],precision_test[4],recall_test[4]],
                                ["KNN",test_score_[3],f1_test[3],precision_test[3],recall_test[3]],
                                ["Decision tree",train_score_[0],f1_test[0],precision_test[0],recall_test[0]],
                                ["Logistic Regression", test_score_[1],f1_test[1],precision_test[1],recall_test[1]],
                                ["SVM", test_score_[2],f1_test[2],precision_test[2],recall_test[2]],
                                ["xgboost",test_score_[5],f1_test[5],precision_test[5],recall_test[5]],
                                ["Voting",test_score_[6],f1_test[6],precision_test[6],recall_test[6]]
                                
                               
                                ], 
                          columns=['Model', 'Testing Accuracy %'
                                  ,'f1 test %','Precision test %','Recall test %'])
results_df1


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # Test plots

# In[83]:



plt.figure(figsize=(15,15))
plt.subplot(2,2,1)
plt.title("Testing Accuracy  plot of algrothims",fontweight="bold")


plt.ylabel("Algorithms",fontweight="bold")
plt.xlabel("Accuracy",fontweight="bold")
plt.xlim(80,95)
plt.barh(algos_names,test_score_,color=['r','b','g','y','m'],height=0.5,edgecolor="k",linewidth=2,hatch='/');



plt.subplot(2,2,2)
plt.title(" f1 test score plot of algrothims",fontweight="bold")

plt.xlim(80,95)
plt.ylabel("Algorithms ",fontweight="bold")
plt.xlabel("f1 score",fontweight="bold")

plt.barh(algos_names,f1_test,color=['r','b','g','y','m'],height=0.5,edgecolor="k",linewidth=2,hatch='/');



plt.subplot(2,2,3)
plt.title(" precision test score plot of algrothims",fontweight="bold")


plt.ylabel("Algorithms ",fontweight="bold")
plt.xlabel("precision",fontweight="bold")
plt.xlim(80,95)
plt.barh(algos_names,precision_test,color=['r','b','g','y','m'],height=0.5,edgecolor="k",linewidth=2,hatch='/');


plt.subplot(2,2,4)
plt.title(" recall test score plot of algrothims",fontweight="bold")


plt.ylabel("Algorithms ",fontweight="bold")
plt.xlabel("recall",fontweight="bold")
plt.xlim(80,95)
plt.barh(algos_names,recall_test,color=['r','b','g','y','m'],height=0.5,edgecolor="k",linewidth=2,hatch='/');


# In[ ]:





# In[84]:


def feature_imp(dataset, model):
    fi = pd.DataFrame()
    fi["feature"] = dataset.columns
    fi["importance"] = model.feature_importances_
    return fi.sort_values(by="importance", ascending=True)


# In[85]:


feature_imp(X, rand_forest).plot(kind='barh', figsize=(12,7), legend=False)
plt.show()


# In[ ]:





# In[86]:


plt.figure(figsize=(12,12))
def plot_accuracy_algos_ht_no_ht(param1,colors,xlabal):
    for i in range(1,5):
        plt.subplot(2,2,i)
        plt.title(columns_result_csv[i-1]+"  comparsion ht and no ht",fontweight="bold")
        plt.ylabel("score",fontweight="bold")
        plt.xlabel(list(xlabal))
        
        plt.plot([0,2,4,6,8,10,12],results_df[param1[i-1]],label=" NO HT",linewidth="3",marker='<',color=colors[i-1])
        plt.plot([0,2,4,6,8,10,12],results_df1[param1[i-1]],label=" HT",linewidth="3",marker='*',color=colors[i])
        plt.legend()

colors=['r','g','b','c','m']# save colors in list , used in above loop iterativeliy
columns_result_csv=['Testing Accuracy %','f1 test %','Precision test %','Recall test %']
xlabal=['Rnf',' KNN','Dt','Logreg','SVM','xgboost',' Voting ']

plot_accuracy_algos_ht_no_ht(columns_result_csv,colors,xlabal) 
#passing these params above in function "Testing Accuracy %","f1 test %","Precision test %","Recall test %"

#plt.savefig("accurcaycomp.png")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




