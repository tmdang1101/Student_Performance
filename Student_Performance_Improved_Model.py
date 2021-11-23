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
#print(dt.head)         ## inspect the data

n = dt.shape[0]         ## number of rows / sample size
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




### Sample Statistics
#print(dt.loc[:, ['gpa','reading','math','composite','ACTcomposite']].describe())
#print(dt.loc[:, ['encoded_gender','encoded_level','encoded_highschool']].describe())





### Graphs
#plt.scatter(dt['composite'], dt['ACTcomposite'])
#plt.xlabel('SAT Composite')
#plt.ylabel('ACT Composite')

plt.xlim(0,4)

plt.scatter(dt['gpa'], dt['ACTcomposite'])
m, b = np.polyfit(dt['gpa'], dt['ACTcomposite'], 1)
#print(m)
plt.plot(m*dt['gpa']+b)
plt.xlabel('GPA')
plt.ylabel('ACT Composite')
plt.show()

#ols = ss.linregress(dt['gpa'], dt['ACTcomposite'])
#print(ols[0])


#####

# Add Groups for Fail, Pass, and Succeed in the Graph

#####



### Model with best accuracy
dx = dt.loc[:, ['gpa','encoded_gender','ACTcomposite']]

#print(dx.head)
tt = dt['STAT2103']    





### Do SVM
# split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(dx, tt, test_size = 0.25)

# =============================================================================
# X_train = dx[:600]
# y_train = tt[:600]
# X_test = dx[600:]
# y_test = tt[600:]
# =============================================================================

svmModel = svm.SVC()
svmModel.fit(X_train, y_train)
#print(svmModel.score(X_test, y_test))      # accuracy score





### Prediction
print(svmModel.predict([[2.33,1,25]]))





### Create histogram of accuracy scores
preds = []

for i in range(100):
    #X_train, X_test, y_train, y_test = train_test_split(dx, tt, test_size = 0.25)
    #svmModel = svm.SVC()
    #svmModel.fit(X_train, y_train)
    preds.append(svmModel.score(X_test, y_test))

plt.hist(preds)
plt.xlabel('Accuracy Score')
plt.ylabel('Frequency')
plt.show()

print(np.mean(preds))       # take average of the accuracy scores