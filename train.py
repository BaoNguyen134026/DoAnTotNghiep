# import scikit-learn dataset library
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn import svm
from sklearn import metrics
import numpy as np
import pickle
# import pandas
data = np.load("sp/data.npy")
data = np.reshape(data, (len(data),15*3))
# print(data[0][50])


# print(data[0].shape)
target = np.load("sp/target.npy")

X_train, X_test, Y_train, Y_test = train_test_split(data, target, test_size=0.3, random_state=109)
# Fit the model on training set
model = LogisticRegression()
model.fit(X_train, Y_train)
# save the model to disk
filename = 'sp/traindongtac.sav'
pickle.dump(model, open(filename, 'wb'))
 
# some time later...
 
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
a = loaded_model.predict([data[100]])

# print(a)
# print([target[100]])
# result = loaded_model.score(X_test, Y_test)
# print(result)