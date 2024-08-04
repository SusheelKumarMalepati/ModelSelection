#importing dependencies
from modelselection import compare_models_crossvalidation
from modelselection import ModelSelection
import numpy as np
import pandas as pd
data=pd.read_csv('./train.csv')
#handling missing values
#dropping cabin column since it is not useful for prediction
data=data.drop(columns='Cabin',axis=1)

#replace with mean age in missing age values
data['Age'].fillna(data['Age'].mean(),inplace=True)
#replacing missing values of embark with mode value
#finding mode
mode=data['Embarked'].mode()[0]
#replacing missing values
data['Embarked'].fillna(mode,inplace=True)
data.replace({'Sex':{'male':0,'female':1}},inplace=True)
data.replace({'Embarked':{'S':0,'C':1,'Q':2}},inplace=True)
#seperating features and target
x=data.drop(['PassengerId','Name','Ticket','Survived'],axis=1)
y=data['Survived']
x=np.asarray(x)
y=np.asarray(y)
compare_models_crossvalidation(x,y)
print('***************************************************************')
print('***************************************************************')
print('***************************************************************')
ModelSelection(x,y)