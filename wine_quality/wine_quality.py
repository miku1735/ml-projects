#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 31 02:58:56 2018

@author: mayank
"""

import sys
import numpy
import pandas
import matplotlib
import seaborn
import scipy
###########################  Finding Versions     ###############################################################

print("\n\n\nVERSION DISCRIPTIONS:-\n")
print("python   :    version:- ",format(sys.version))
print("numpy    :    version:- ",format(numpy.__version__))
print("pandas   :    version:- ",format(pandas.__version__))
print("matplotlib :  version:- ",format(matplotlib.__version__))
print("seaborn  :    version:- ",format(seaborn.__version__))
print("scipy    :    version:- ",format(scipy.__version__))


##########################      Importing nessary modules       ##################################################


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


###########################     Cleaning data         #############################################################


wine = pd.read_csv('winequality-red.csv')
bins = (2, 6.5, 8)          ###########    giving the range for no and yes respectively     #######################
group_names = ['bad', 'good']
wine['quality'] = pd.cut(wine['quality'], bins = bins, labels = group_names)

label_quality = LabelEncoder()        #############      here LabelEncoder()     {good = 1, bad = 0}     ##########
wine['quality'] = label_quality.fit_transform(wine['quality'])

X = wine.drop('quality', axis = 1)
y = wine['quality']

##############################       Train and Test splitting of data            ################################### 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

#############          Applying Standard scaling to get optimized {result range 0 to 1}           ##################
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

###################################       RandomForestClassifier           #########################################

print("\n\n\n\n\t\t\tRandomForestClassifier\n")
rfc = RandomForestClassifier(n_estimators=200,random_state=1)
rfc.fit(X_train, y_train)
pred_rfc = rfc.predict(X_test)

print(classification_report(y_test, pred_rfc))
print(confusion_matrix(y_test, pred_rfc))

###################################         SGDClassifier           #################################################

print("\n\n\n\n\t\t\tSGDClassifier\n")
sgd = SGDClassifier(penalty=None)
sgd.fit(X_train, y_train)
pred_sgd = sgd.predict(X_test)

print(classification_report(y_test, pred_sgd))
print(confusion_matrix(y_test, pred_sgd))

####################################        SVC from SVM      #######################################################

print("\n\n\n\n\t\t\tSVC\n")
svc = SVC()
svc.fit(X_train, y_train)
pred_svc = svc.predict(X_test)

print(classification_report(y_test, pred_svc))
print(confusion_matrix(y_test, pred_svc))

print("\n\n\n\n\n\n\n")

#####################################    complete      ###############################################################