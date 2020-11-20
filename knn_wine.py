import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier


#load data from the csv file
data_wine = pd.read_csv('data_wine_after_processing1.csv')
data_wine=data_wine.sample(frac=1)
# labels
data = data_wine.values
    #print(array)
    #print(type(array))
X=data[:,0:11]
Y=data[:,12]
    #print(X)
    #print(Y)

#train and test split

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.1,random_state=0)
    #print(X_train.shape)
    #print(X_test.shape)
    #print(Y_train.shape)
    #print(Y_test.shape)

#Initialize Gaussian Naive Bayes

clf=KNeighborsClassifier(n_neighbors=3)


#use Cross_validation for this data to validation
kfold=KFold(n_splits=5)
scores=cross_val_score(clf,X_train,Y_train,cv=kfold)
real_scores=scores.mean()*100
print("Accuracy of this model KNeighborsClassification : ",real_scores)

#predict with test set
clf.fit(X_train,Y_train)
pred_clf=clf.predict(X_test)
print("predict tesset: ",pred_clf)