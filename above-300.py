# -*- coding: utf-8 -*-
"""
Created on Sun Jul 15 11:26:22 2018

@author: dell pc
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
dataset = pd.read_csv('odi_cricket_dataset.csv')
X = dataset.iloc[:, [1, 2, 3, 4, 5]].values
y = dataset.iloc[:, -1].values


# encoding categorical data i.e team , opposition, ground
from sklearn.preprocessing import LabelEncoder , OneHotEncoder

label_encoder_X_0 = LabelEncoder()
X[:,0] = label_encoder_X_0.fit_transform(X[:,0])

label_encoder_X_1 = LabelEncoder()
X[:,1] = label_encoder_X_1.fit_transform(X[:,1])

label_encoder_X_2 = LabelEncoder()
X[:,2] = label_encoder_X_2.fit_transform(X[:,2])


onehotencoder_0 = OneHotEncoder(categorical_features= [0,1,2])
X = onehotencoder_0.fit_transform(X).toarray()


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, 130:132] = sc.fit_transform(X_train[:, 130:132])
X_test[:, 130:132] = sc.transform(X_test[:, 130:132])


#fitting logistics Regression to the training set 73.8%

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Fitting Decision tree to the Training set 76.2%

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state= 0)  
classifier.fit(X_train, y_train)


# Fitting SVM to the Training set 66.9%

from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)   
classifier.fit(X_train, y_train)


# Fitting K-NN to the Training set 69.4%

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 20, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)


# Fitting Random Forest to the Training set 75.4%

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy' , random_state = 0 )  
classifier.fit(X_train, y_train)




# Predicting the Test set results 
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#accuracy from model
accuracy = ((cm[0,0] + cm[1,1]) / 118) * 100


