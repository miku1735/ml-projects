# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#################################   importing the dataset ###############################################################

dataset = pd.read_csv('Restaurant_Reviews.tsv',delimiter='\t',quoting = 3) 


#################################   cleaning the text  ##################################################################
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = []

#stopwords.words('english')

for i in range(0,1000):
    review = re.sub('[^a-zA-Z]',' ',dataset["Review"][i])  ### subpart replacing extra variables with space char ########
    review=review.lower()  #########    lower the capitals ##############################################################
    
    review=review.split()
    for j in review:
        if(j in set(stopwords.words('english'))):
            review.remove(j)
    
    
    ####################################   stemming    ##################################################################
    
    ps=PorterStemmer()
    
    review = [ps.stem(words) for words in review ]
    
    ######################################  joining all the words #######################################################
    review1 = ""
    for word in review:
        review1 = review1 + word + " "
    review=review1.strip()
    corpus.append(review)


############################  bag of word model   #######################################################################


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features = 1300)  ######## 1300 is most 1300 frequent variable ###################################
X=cv.fit_transform(corpus).toarray()      
y=dataset.iloc[:,-1].values
rt=cv.get_feature_names()
#rt[:10]
##########################################################################################################################\

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 1)

###################################       RandomForestClassifier           ################################################

print("\n\n\n\n\t\t\tRandomForestClassifier\n")
rfc = RandomForestClassifier(n_estimators=200,random_state=1)
rfc.fit(X_train, y_train)
pred_rfc = rfc.predict(X_test)

print(classification_report(y_test, pred_rfc))
print(confusion_matrix(y_test, pred_rfc))

############################################################################################################################


str=1
while(str):
    print("\n")
    str=input("Enter the user review:-")
    
    review = re.sub('[^a-zA-Z]',' ',str)  ### subpart replacing extra variables with space char ########
    review=review.lower()  #########    lower the capitals ##############################################################
        
    review=review.split()
    for j in review:
       if(j in set(stopwords.words('english'))):
           review.remove(j)
        
        
        ####################################   stemming    ##################################################################
        
    ps=PorterStemmer()
        
    review = [ps.stem(words) for words in review ]
        
    ######################################  joining all the words #######################################################
    review1 = ""
    for word in review:
        review1 = review1 + word + " "
    review=review1.strip()
    corpus.append(review)
    
    from sklearn.feature_extraction.text import CountVectorizer
    cv=CountVectorizer(max_features = 1300)  ######## 1300 is most 1300 frequent variable ###################################
    X=cv.fit_transform(corpus).toarray()      
    y=dataset.iloc[:,-1].values
    x_test=X[-1]
    x_test=x_test.reshape(1,1300)
    ans=rfc.predict(x_test)
    if(ans[0]==1):
        print("\n Thanks for coming and giving positive review.")
    else:
        print("\n We will try our best next time,sorry for inconveniance.")
    str=input("0 for stop 1 for continue")