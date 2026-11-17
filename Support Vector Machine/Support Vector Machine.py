import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
%matplotlib inline 

iris = sns.load_dataset('iris')
sns.pairplot(iris,hue='species')

setosa = iris[iris['species']=='setosa']
sns.kdeplot(
    x=setosa['sepal_width'],
    y=setosa['sepal_length'],
    cmap='plasma',
    fill=True
)

X = iris.drop('species',axis=1)
y = iris['species']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=101)
from sklearn.svm import SVC 
svc_model = SVC()
svc_model.fit(X_train,y_train)
predicti = svc_model.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(predicti,y_test))
print(confusion_matrix(predicti,y_test))
from sklearn.model_selection import GridSearchCV
param_grid = {'C':[0.1,1,10,100],'gamma':[1,0.1,0.01,0.001]}
grid = GridSearchCV(SVC(),param_grid,verbose=2)
grid.fit(X_train,y_train)

grid_pred = grid.predict(X_test)
print(classification_report(grid_pred,y_test))
print(confusion_matrix(grid_pred,y_test))