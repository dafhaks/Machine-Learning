import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

loans = pd.read_csv(r'D:\Machine Learning\Decision Trees\loan_data.csv')
loans.head()

plt.figure(figsize=(10,6))

loans[loans['credit.policy'] == 1]['fico'].hist(
    label='Credit Policy = 1',
    alpha=0.6
)

loans[loans['credit.policy'] == 0]['fico'].hist(
    label='Credit Policy = 0',
    alpha=0.6
)

plt.legend()
plt.show()

plt.figure(figsize=(11,7))
sns.countplot(x='purpose',hue='not.fully.paid',data=loans,palette='Set1')

cat_feats = ['purpose']
final_data = pd.get_dummies(loans,columns=cat_feats,drop_first=True)
final_data.info()

X = final_data.drop('not.fully.paid',axis=1)
y = final_data['not.fully.paid']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=101)

from sklearn.tree import DecisionTreeClassifier 
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)
predicti = dtree.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(predicti,y_test))
print(confusion_matrix(predicti,y_test))