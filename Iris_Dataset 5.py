import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier 
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

#Loading the data
data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

#dividing the dataset into features and labels
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

#splitting the data into 70% training and 30% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

#using Decision Tree Classifier
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

#using Support Vector Machine
svc = SVC()
svc.fit(X_train, y_train)

#predicting the results
y_pred_dt = dt.predict(X_test)
y_pred_svc = svc.predict(X_test)

#finding the accuracy of the predictions
acc_dt = accuracy_score(y_test, y_pred_dt)
acc_svc = accuracy_score(y_test, y_pred_svc)

#printing the accuracy
print(f"Accuracy of Decision Tree Classifier: {round(acc_dt,4)*100} %")
print(f"Accuracy of Support Vector Machine: {round(acc_svc,4)*100} %")
