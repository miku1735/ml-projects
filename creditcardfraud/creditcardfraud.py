import sys
import numpy
import pandas
import matplotlib
import seaborn
import scipy

###################################           versions       ###############################################


print("\nVERSION DISCRIPTIONS:-\n")
print("python   :    version:- ",format(sys.version))
print("numpy    :    version:- ",format(numpy.__version__))
print("pandas   :    version:- ",format(pandas.__version__))
print("matplotlib :  version:- ",format(matplotlib.__version__))
print("seaborn  :    version:- ",format(seaborn.__version__))
print("scipy    :    version:- ",format(scipy.__version__))


#########################        importing only needed packages            #################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



#############################       loading the csv file       ##############################################

print("\n\n")
data=pd.read_csv("creditcard.csv")

##############################  explring the dataset    #####################################################

print("DATA DISCRIPTIONS:-")
print("\n Columns\n")
print(data.columns)

print (" \n\nclass here gona be a 0 AND 1\n 0 :- Valid card transaction \n 1 :- Invalid card tarnsaction\n\n")
print("\ndata has total (row,column):-",data.shape)

print(data.describe())

print("\n\n              RATIO BETWEEN 0 AND 1 ")
#m=data.iloc[:,30]
#m.hist(figsize=(5,5))
#plt.show()


###############################################################################################################


fraud = data[data['Class'] == 1]
valid = data[data['Class'] == 0]

outlier_fraction=len(fraud) / float(len(valid))

print("outlier_fraction:-",outlier_fraction)
print("Fraud cases:-",format(len(fraud)))
print("Valid cases:-",format(len(valid)))

#################################### to test is any column is there for which the is dependent more  ###########

cormat=data.corr()
fig=plt.figure(figsize=(5,5))

######################               seaborn for sns heat map            ########################################

sns.heatmap(cormat,square = True)
plt.show()
x=data.iloc[:,:-1].values
y=data.iloc[:,-1].values


###############    exploring complete         ####################################################################
##################################################################################################################
#############  we use isolation forest algo and local outlier factor algo for anamoly detection ##################
     

from sklearn.metrics import classification_report,accuracy_score
from sklearn.ensemble import IsolationForest   #######3randomly select the values   ##############################
from sklearn.neighbors import LocalOutlierFactor ########### unsupervised outlier detection method ###############


################## defining the dictionary for the functions  ####################################################

classifiers = {
        "Local Outlier Factor":LocalOutlierFactor(n_neighbors=20,contamination = outlier_fraction),
        "Isolation Forest": IsolationForest(max_samples=len(x),contamination = outlier_fraction,random_state = 1)
        } 


###############       fitintg to the model            #############################################################



for clf_name,clf in classifiers.items():
    if(clf_name=="Local Outlier Factor"):
        
        y_pred=clf.fit_predict(x)
        scores_pred = clf.negative_outlier_factor_
    else:
        
        clf.fit(x)
        scores_pred = clf.decision_function(x)
        y_pred = clf.predict(x)
        
        
    y_pred[y_pred == 1]=0
    y_pred[y_pred == -1]=1
    
    n_errors = str((y!=y_pred).sum())
    print("\n\n\n\n\n")
    print('%s:%s'%(format(clf_name),(format(n_errors))))
    print(accuracy_score(y,y_pred))
    print(classification_report(y,y_pred))
    
##############################################   complete    ##########################################################