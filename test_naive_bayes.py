import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


#load data
data_wine = pd.read_csv('data_wine_after_processing1.csv')
data_wine=data_wine.sample(frac=1) #xáo chon du lieu
# labels
data = data_wine.values#chuyen tu data frame ve dang ma tran
    #print(array)
    #print(type(array))
X=data[:,0:11] #lua chon cac bien doc lap
Y=data[:,12]# thuoc tinh can du doan
    #print(X)
    #print(Y)

#train and test split
#chia tep du lieu thanh 2 phan test set và train set test set ti le testset 0.1
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.1,random_state=0)

#khoi tao bo phan lop Gaussian Naive Bayes

clf=GaussianNB()

#dung Cross_validation cho kiem dinh bo phan lop
kfold=KFold(n_splits=5)
scores=cross_val_score(clf,X_train,Y_train,cv=kfold)
real_scores=scores.mean()*100
print("Accuracy of this model Naive Bayes: ",real_scores)

#du doan ket qua voi test set
clf.fit(X_train,Y_train)
pred_clf=clf.predict(X_test)
print("predict tesset: ",pred_clf)












