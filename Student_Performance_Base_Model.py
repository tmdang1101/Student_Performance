import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.cluster.vq import kmeans, vq, whiten
from sklearn.model_selection import train_test_split
import sklearn as skl
from sklearn import svm
from numpy import arange

import scipy.stats as ss




### Load data
url = "https://courses.kvasaheim.com/stat195/project/forsbergCollege.csv"
dt = pd.read_csv(url)
#print(dt.head)     ## inspect the data

n = dt.shape[0]     ## number of rows / sample size
#print(n)           




### Encode categorical data into numerical values
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
gender_encoded = le.fit_transform(dt['gender'])
dt['encoded_gender'] = gender_encoded
#print(gender_encoded)

level_encoded = le.fit_transform(dt['level'])
dt['encoded_level'] = level_encoded
#print(level_encoded)
    
highschool_encoded = le.fit_transform(dt['highschool'])
dt['encoded_highschool'] = highschool_encoded
#print(highschool_encoded)

#encoded_gender: 0 - Female, 1 - Male
#encoded_level: 0 - Freshman, 1 - Junior, 2 - Senior, 3 - Sophomore
#encoded_highschool: 0 - Home School, 1 - Private High School, 2 - Public High School, 3 - Transfer





### Sample Statistics
#print(dt.loc[:, ['gpa','reading','math','composite','ACTcomposite']].describe())
#print(dt.loc[:, ['encoded_gender','encoded_level','encoded_highschool']].describe())
#print(dt.loc[:, ['gender','level','highschool','STAT2103']].describe())
#print(dt['gender'].value_counts())
#print(dt['level'].value_counts())
print(dt['highschool'].value_counts())
print(dt['STAT2103'].value_counts())

#plt.hist()
#plt.hist(dt['gpa'])
#plt.hist(dt['reading'])
#plt.hist(dt['math'])
#plt.hist(dt['composite'])
#plt.hist(dt['ACTcomposite'])

#plt.xlabel('ACT Composite')
#plt.ylabel('Frequency')

y = np.array([371,111,161])
myLabels = ['Pass','Fail','Succeed']
plt.pie(y, labels = myLabels, startangle = 90, autopct='%.2f')
plt.xlabel('Proportion of Student Performance in STAT 2103')
plt.show()







### Base Model - includes all independent variables 
#dx = dt.loc[:, ['gpa','reading','math','composite','encoded_gender','encoded_level','encoded_highschool','ACTcomposite']]
#print(dx.head)         # checks the first few values of dx
#tt = dt['STAT2103']     # target / dependent variable





### Do SVM
# split the dataset into train and test sets
#X_train, X_test, y_train, y_test = train_test_split(dx, tt, test_size = 0.25)
#svmModel = svm.SVC()
#svmModel.fit(X_train, y_train)
#print(svmModel.score(X_test, y_test))      # accuracy score





### Prediction
#print(svmModel.predict([[2.33,750,720,1470,1,2,1,25]]))





### Create histogram of accuracy scores
# =============================================================================
# preds = []
# 
# for i in range(100):
#     X_train, X_test, y_train, y_test = train_test_split(dx, tt, test_size = 0.25)
#     svmModel = svm.SVC()
#     svmModel.fit(X_train, y_train)
#     
#     preds.append(svmModel.score(X_test, y_test))
# 
# #plt.hist(preds)
# #plt.xlabel('Accuracy Score')
# #plt.ylabel('Frequency')
# #plt.show()
# 
# print(np.mean(preds))       # take average of the accuracy scores
# =============================================================================
