import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
%matplotlib inline 

ad_data = pd.read_csv(r'D:\Machine Learning\Logistic Regression\advertising.csv')
ad_data.head()
sns.jointplot(x='Age',y='Area Income',data=ad_data)

from sklearn.model_selection import train_test_split
X = ad_data[['Daily Time Spent on Site','Age','Area Income','Daily Internet Usage','Male']]
y = ad_data['Clicked on Ad']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=101)

from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
prediction = logmodel.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(prediction,y_test))
print(confusion_matrix(prediction,y_test))