#importing dependencies
from modelselection import compare_models_crossvalidation
from modelselection import ModelSelection
import numpy as np
import pandas as pd
#loading data
data=pd.read_csv('./heart.csv')
#seperating features and target
x=data.drop(columns='target',axis=1)
y=data['target']
x=np.asarray(x)
y=np.asarray(y)
compare_models_crossvalidation(x,y)
print('***************************************************************')
print('***************************************************************')
print('***************************************************************')
ModelSelection(x,y)
