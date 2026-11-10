import numpy as np
import seaborn as sns 
import pandas as pd
import matplotlib.pyplot as plt 
%matplotlib inline

df = pd.read_csv(r'D:\Machine Learning\K Nearest Neighbour\KNN_Project_Data')
df.head()
sns.pairplot(df,hue='TARGET CLASS', palette='coolwarm')

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df.drop('TARGET CLASS',axis=1))
scaled_feat = scaler.transform(df.drop('TARGET CLASS',axis=1))
df_feat = pd.DataFrame(scaled_feat,columns=df.columns[:-1])
df_feat.head()

X = df_feat
y = df['TARGET CLASS']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=101)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
pred = knn.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(pred,y_test))
print(confusion_matrix(pred,y_test))

error_rate = []
for i in range (1,40):
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))
    

plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue',linestyle='--',marker='o',)
plt.xlabel('K')
plt.ylabel("ERROR_RATE")

knn = KNeighborsClassifier(n_neighbors=30)
knn.fit(X_train,y_train)
prediction = knn.predict(X_test)
print(classification_report(prediction,y_test))
print(confusion_matrix(prediction,y_test))

