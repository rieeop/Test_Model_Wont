import pandas as pd
import tensorflow
import keras
import os
import glob
import numpy as np
import sklearn
from sklearn import linear_model, preprocessing
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style
from sklearn import metrics
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv("BEAUZ & Momo - Won_t Look Back -.csv", sep=",")
data = shuffle(data, random_state=22)



le = preprocessing.LabelEncoder()
Time = le.fit_transform(list(data["Time (s)"]))
AccelerationX = le.fit_transform(list(data["Acceleration x (m/s^2)"]))
AccelerationY = le.fit_transform(list(data["Acceleration y (m/s^2)"]))
AccelerationZ = le.fit_transform(list(data["Acceleration z (m/s^2)"]))
AbsoluteAcceleration = le.fit_transform(list(data["Absolute acceleration (m/s^2)"]))



X = list(zip(Time, AccelerationX, AccelerationY, AccelerationZ))
y = list(AbsoluteAcceleration)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

"""
best = 0
for _ in range(30):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

    model = KNeighborsClassifier(n_neighbors=3)

    model.fit(x_train, y_train)
    acc = model.score(x_test, y_test)
    print(acc)

    if acc > best:
        best = acc

    with open("BEAUZModel.pickle","wb") as f:
        pickle.dump(model, f)"""

pickle_in = open("BEAUZModel.pickle", "rb")
model = pickle.load(pickle_in)

predicted= model.predict(x_test)
for x in range(len(predicted)):
    print(predicted[x], x_test[x], y_test[x])



"""
style.use("ggplot")
pyplot.scatter(data["Time (s)"],data["Absolute acceleration (m/s^2)"])
pyplot.xlabel("Time (s)")
pyplot.ylabel("Final Grade")
pyplot.show()"""