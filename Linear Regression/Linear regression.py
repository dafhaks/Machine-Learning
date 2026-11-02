import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
%matplotlib inline 
customers = pd.read_csv(r'C:\Users\muham\Downloads\Refactored_Py_DS_ML_Bootcamp-master\11-Linear-Regression\Ecommerce Customers')

sns.jointplot(data=customers,x='Time on Website',y='Yearly Amount Spent')
sns.jointplot(data=customers,x='Time on App',y='Yearly Amount Spent')
sns.lmplot(x='Length of Membership',y='Yearly Amount Spent',data=customers)

y = customers['Yearly Amount Spent']
X = customers[['Avg. Session Length', 'Time on App',
       'Time on Website', 'Length of Membership']]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=101)

from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)
prediction = lm.predict(X_test)
from sklearn import metrics 
print(metrics.mean_absolute_error(prediction,y_test))
print(metrics.mean_squared_error(prediction,y_test))
print(np.sqrt(metrics.mean_squared_error(prediction,y_test)))