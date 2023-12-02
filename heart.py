# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 08:58:53 2022

@author: KTong
"""

import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import missingno as msno
import scipy.stats as ss
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score,classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV


#%% FUNCTIONS
def cont_plot(num_features):
    '''
    Creates plot for numerical data.

    Parameters
    ----------
    num_features : list
        Column names of numerical features.

    Returns
    -------
    seaborn.distplot().

    '''
    for i in num_features:
        plt.figure()
        sns.distplot(df[i])
        plt.show()

def cat_plot(cat_features):
    '''
    Creates plot for categorical data.

    Parameters
    ----------
    cat_features : list
        Column names of numerical features.

    Returns
    -------
    seaborn.countplot().

    '''
    for i in cat_features:
        plt.figure()
        sns.countplot(df[i])
        plt.show()

def cramers_corrected_stat(confusion_matrix):
    """ calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher, 
        Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))

def con_mat_plot(y_true,y_pred):
    '''
    Generates confusion matrix.

    Parameters
    ----------
    y_true : ndarray
        Original labels of target column.
    y_pred : ndarray
        Predicted labels of target column of model.

    Returns
    -------
    Confusion matrix plot.

    '''
    cm=confusion_matrix(y_true,y_pred)
    disp=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=['Positive','Negative'])
    disp.plot(cmap=plt.cm.Reds)
    plt.show()

#%% STATICS
DATA_FILE_PATH=os.path.join(os.getcwd(),'dataset','heart.csv')
TEST_DATA_PATH=os.path.join(os.getcwd(),'dataset','heart_test.csv')
PIPELINE_PICKLE_PATH=os.path.join(os.getcwd(),'model','pipeline_heart.pkl')
MODEL_PICKLE_PATH=os.path.join(os.getcwd(),'model','model_heart.pkl')

#%% DATA LOADING
df=pd.read_csv(DATA_FILE_PATH)

#%% DATA INSPECTION
# Identify numeric / categorical features
column_names=df.columns
num_data=['age','trtbps','chol','thalachh','oldpeak','caa']
cat_data=['sex','cp','fbs','restecg','exng','slp','thall','output']

# Check for NaNs and duplicates
msno.matrix(df)
msno.bar(df)
df[df['thall']==0]

df.duplicated().sum() 
df[df.duplicated()] 

# Check for errorneous observation in numeric features
df.boxplot(rot=90)
df[num_data].describe().T

# 2 values present in dataset, dataset has no abnormal values,
# 1 duplicate observation is observed

#%% DATA VISUALISATION
cont_plot(num_data)
cat_plot(cat_data)

# Data is balanced

#%% DATA CLEANING
# Remove duplicates
df_clean=df.drop_duplicates()

# Low number of duplicated row present in balanced dataset hence removed

# Impute NULL in 'thall'
df_clean['thall'].replace(to_replace=0,value=np.median(df_clean['thall']),
                          inplace=True)

#%% FEATURE SELECTION
# Check correlation of numeric features to target
lr=LogisticRegression()

for i in num_data:
    lr.fit(np.expand_dims(df_clean[i],axis=-1),df_clean['output'])
    print('{}: R^2 is {}'.format(i,lr.score(np.expand_dims(df_clean[i],axis=-1),df_clean['output'])))

# age(0.62),trtbps(0.58),chol(0.53),thalachh(0.70),oldpeak(0.69),caa(0.74)
# remove 'age','trtbps' and 'chol' since accuracy is low, inferring low 
# correlation to output

# Check correlation of categorical features to target
for i in cat_data:
    con_mat=pd.crosstab(df_clean[i], df_clean['output']).to_numpy()
    print('{}: accuracy is {}'.format(i,cramers_corrected_stat(con_mat)))

# sex(0.27),cp(0.51),fbs(0.00),restecg(0.16),exng(0.43),slp(0.39),thall(0.52)
# remove 'sex','fbs','restecg','exng',and 'slp' due to low correlation

# Finalised features
x=df_clean[['thalachh','oldpeak','caa','cp','thall']]
y=df_clean['output']

#%% PREPROCESSING
# Create train and test dataset
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,
                                               random_state=123,
                                               stratify=y)

#%% PIPELINE BUILDING
# Logistic Regression pipelines
pl_std_lr=Pipeline([('StandardScaler', StandardScaler()),
                    ('LogisticClassifier', LogisticRegression())])

pl_mms_lr=Pipeline([('MinMaxScaler', MinMaxScaler()),
                    ('LogisticClassifier', LogisticRegression())])

# DecisionTree pipelines
pl_std_dt=Pipeline([('StandardScaler', StandardScaler()),
                    ('DTClassifier', DecisionTreeClassifier())])

pl_mms_dt=Pipeline([('MinMaxScaler', MinMaxScaler()),
                    ('DTClassifier', DecisionTreeClassifier())])

# RandomForest pipelines
pl_std_rf=Pipeline([('StandardScaler', StandardScaler()),
                    ('RFClassifier', RandomForestClassifier())])

pl_mms_rf=Pipeline([('MinMaxScaler', MinMaxScaler()),
                    ('RFClassifier', RandomForestClassifier())])

# KNN pipelines
pl_std_knn=Pipeline([('StandardScaler', StandardScaler()),
                     ('KNNClassifier', KNeighborsClassifier())])

pl_mms_knn=Pipeline([('MinMaxScaler', MinMaxScaler()),
                     ('KNNClassifier', KNeighborsClassifier())])
# SVM pipelines
pl_std_svc=Pipeline([('StandardScaler', StandardScaler()),
                     ('SVClassifier', SVC())])

pl_mms_svc=Pipeline([('MinMaxScaler', MinMaxScaler()),
                     ('SVClassifier', SVC())])

pipelines=[pl_std_lr,pl_mms_lr,pl_std_dt,pl_mms_dt,
           pl_std_rf,pl_mms_rf,pl_std_knn,pl_mms_knn,
           pl_std_svc,pl_mms_svc]
             
#%% PIPELINE EVALUATION
for i in pipelines:
    i.fit(x_train,y_train)

best_accuracy=0

dict_pl={0:'Std+LR',1:'MMS+LR',2:'Std+DT',3:'MMS+DT',
         4:'Std+RF',5:'MMS+RF',6:'Std+KNN',7:'MMS+KNN',
         8:'Std+SVC',9:'MMS+SVC'}

# Find out best model with MinMax/ Standard Scaler approach
for i, model in enumerate(pipelines):
    print(dict_pl[i])
    print(model.score(x_test,y_test))
    if model.score(x_test,y_test) > best_accuracy:
        best_model=model
        best_accuracy = model.score(x_test,y_test)
        best_approach=dict_pl[i]

print('Best model: {} with accuracy of {}'.format(best_approach,best_accuracy))
print(best_model)

# Best model: Std+LR with accuracy of 0.87

#%% FINETUNE PIPELINE
# Define pipeline
pipeline_LR=Pipeline([('StandardScaler', StandardScaler()),
                      ('LogisticClassifier', LogisticRegression())])

# Define parameters to test
grid_param=[{'LogisticClassifier':[LogisticRegression()],
             'LogisticClassifier__penalty':['none','l2','l1','elasticnet'],
             'LogisticClassifier__C':np.arange(0,2,0.1)}]

grid_search=GridSearchCV(pipeline_LR,grid_param,cv=5,verbose=2)
best_LR_pipeline=grid_search.fit(x_train,y_train)
best_LR_pipeline.best_params_
best_LR_pipeline.best_score_
best_LR_pipeline.best_estimator_
# Best parameter for Logistic Regression is : C=0.4, 
#                                             penalty='l2'

#%% PIPELINE EVALUATION
y_pred=best_LR_pipeline.predict(x_test)

print(classification_report(y_test,y_pred))
print(accuracy_score(y_test,y_pred))
con_mat_plot(y_test,y_pred)

#%% PIPELINE EXPORT
with open(PIPELINE_PICKLE_PATH,'wb') as file:
    pickle.dump(best_LR_pipeline,file)

#%% FINALISED MODEL
best_pipeline_LR=Pipeline([('StandardScaler', StandardScaler()),
                           ('LogisticClassifier', LogisticRegression(C=0.4))])

best_LR_model=best_pipeline_LR.fit(x_train,y_train)

# Test model with test dataset
df_test=pd.read_csv(TEST_DATA_PATH)
x_df_test=df_test.loc[:,['thalachh','oldpeak','caa','cp','thall']]
y_df_test=df_test['output']

y_df_test_pred=best_LR_model.predict(x_df_test)

# Check accuracy of model
print(accuracy_score(y_df_test,y_df_test_pred))
print(classification_report(y_df_test,y_df_test_pred))
con_mat_plot(y_df_test,y_df_test_pred)

#%% MODEL EXPORT
with open(MODEL_PICKLE_PATH,'wb') as file:
    pickle.dump(best_LR_model,file)
