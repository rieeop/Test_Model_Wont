import tensorflow
import keras
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

import pickle
import matplotlib.pyplot as pyplot
from matplotlib import style


data = pd.read_csv("BEAUZ & Momo - Won_t Look Back -.csv", sep=",")
data = shuffle(data, random_state=22)


data = np.genfromtxt("BEAUZ & Momo - Won_t Look Back -.csv", dtype=float, delimiter=',', names=True)


predict = "Time (s)"

X = np.array(data.[predict,1])
y = np.array(data[predict])



x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)




linear = linear_model.LinearRegression()

linear.fit(x_train, y_train)
acc = linear.score(x_test, y_test)
print(acc)