import pandas as pd
#importing models
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import cross_val_score
models=[LogisticRegression(max_iter=1000),LinearRegression(),Lasso(),Ridge(),SVC(),KNeighborsClassifier(),RandomForestClassifier(random_state=0),DecisionTreeClassifier(),MLPClassifier(max_iter=1000)]
def compare_models_crossvalidation(feature,target):
  for model in models:
    csv_score=cross_val_score(model,feature,target,cv=5)
    mean_acc=sum(csv_score)/len(csv_score)
    mean_acc=mean_acc*100
    mean_acc=round(mean_acc,2)
   # print('cross validation accuracies for the ',model,' is ',csv_score)
    print('accuracy score of the ',model,' is ',mean_acc)
    print('---------------------------------------------------------------')

models_list=[LogisticRegression(max_iter=10000),SVC(),KNeighborsClassifier(),RandomForestClassifier(random_state=0),DecisionTreeClassifier(),MLPClassifier(max_iter=10000)]
#dictionary for hyperparameter values for models
hyperparameters_dict={
    'log_reg_hyperparameters':{
        'C':[1,5,10,20]
    },
    'svc_hyperparameters':{
        'kernel':['linear','poly','rbf','sigmoid'],
        'C':[1,5,10,20]
    },
    'KNN_hyperparameters':{
        'n_neighbors':[3,5,10]
    },
    'random_forest_hyperparameters':{
        'n_estimators':[10,20,50,100]
    },
    'dec_tree_hyperparameters':{
        'max_leaf_nodes':[5,20,30,50]
    },
    'MLP_clas_hyperparameters':{
        'hidden_layer_sizes':[20,50,80,100,130,150]
    }
}
from sklearn.model_selection import GridSearchCV
def ModelSelection(feature,target):
  result=[]
  i=0
  for model in models_list:
    model_keys=list(hyperparameters_dict.keys())
    key = model_keys[i]
    params=hyperparameters_dict[key]
    i+=1
    print(model)
    print(params)
    classifier=GridSearchCV(model,params,cv=5)
    #fitting data to classifier
    classifier.fit(feature,target)
    result.append({
        'model used' : model,
        'highest score' : classifier.best_score_,
        'best hyperparameters' : classifier.best_params_
    })
    print('---------------------------------------------------------------')
  for i in range (len(models_list)):
    print(result[i])
  