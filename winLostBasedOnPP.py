# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 11:43:46 2018

@author: dell pc
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jul 15 09:02:08 2018

@author: dell pc
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('odi_cricket_final.csv')

X = dataset.iloc[:, [1, 2, 3, 4, 5]].values
y = dataset.iloc[:, -1].values


"""      #already encoded the result in data
#encoding the result
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
label_encoder_y = LabelEncoder()
y = label_encoder_y.fit_transform(y)"""


# encoding categorical data i.e team , opposition, ground

from sklearn.preprocessing import LabelEncoder , OneHotEncoder


label_encoder_X_0 = LabelEncoder()
X[:,0] = label_encoder_X_0.fit_transform(X[:,0])

label_encoder_X_1 = LabelEncoder()
X[:,1] = label_encoder_X_1.fit_transform(X[:,1])

label_encoder_X_2 = LabelEncoder()
X[:,2] = label_encoder_X_2.fit_transform(X[:,2])


onehotencoder_0 = OneHotEncoder(categorical_features= [0])
X = onehotencoder_0.fit_transform(X).toarray()

onehotencoder_1 = OneHotEncoder(categorical_features= [19])
X = onehotencoder_1.fit_transform(X).toarray()

onehotencoder_2 = OneHotEncoder(categorical_features= [38])
X = onehotencoder_2.fit_transform(X).toarray()



# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)





# Fitting K-NN to the Training set 53.3%
'''
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)'''


# Fitting SVM to the Training set 57.6%
'''
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)   
classifier.fit(X_train, y_train)'''


# Fitting Decision tree to the Training set 50%
'''
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state= 0)  
classifier.fit(X_train, y_train)'''




# Fitting Random Forest to the Training set 51.6%
'''
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy' , random_state = 0 )  
classifier.fit(X_train, y_train)'''


#fitting logistics Regression to the training set 57.6%

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)



# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)




