from imageio import imread, imwrite
import pickle
import numpy as np
import tensorflow as tf
import keras
from keras.models import load_model
import keras.backend as K
from scipy.spatial.distance import cosine
import timeit
import falconn

from lib.lib_knn import *
from lib.knn_attack import *

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
from keras.backend.tensorflow_backend import set_session
set_session(sess)

np.random.seed(1234)

from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train[:, :, :, np.newaxis].astype(np.float32) / 255.
X_test = X_test[:, :, :, np.newaxis].astype(np.float32) / 255.
X_train_nm = X_train/np.sqrt(np.sum(X_train**2, axis=(1, 2, 3), keepdims=True))
X_test_nm = X_test/np.sqrt(np.sum(X_test**2, axis=(1, 2, 3), keepdims=True))
X_train_nm = X_train_nm.reshape(-1, 784)
X_test_nm = X_test_nm.reshape(-1, 784)

# Randomly chosen calibrate set 75 samples from each class
ind_cal = np.zeros((750, ), dtype=np.int32)
for i in range(10):
    ind = np.where(y_test == i)[0]
    np.random.shuffle(ind)
    ind_cal[i*75 : (i + 1)*75] = ind[:75]
ind_test = np.arange(len(X_test), dtype=np.int32)
ind_test = np.setdiff1d(ind_test, ind_cal)

X_test_nm = X_test_nm[ind_test]
y_test = y_test[ind_test]

# nn = np.zeros((len(X_test_nm), 75), dtype=np.int32)
# k_dist = 0
# for i, x in enumerate(X_test_nm):
#     if i % 200 == 0:
#         print(i)
#     dist = np.sum((X_train_nm - x)**2, axis=1)
#     ind = np.argsort(dist)
#     k_dist += dist[74]
#     nn[i] = ind[:75]

# pickle.dump(nn, open("nn_75.p", "wb"))

nn = pickle.load(open("nn_75.p", "rb"))
print(nn[:, -1])
print(X_train_nm[nn[:, -1]].shape)
print(X_test_nm.shape)
dist = np.sqrt(np.sum((X_train_nm[nn[:, -1]] - X_test_nm)**2, axis=1))
# dist = np.sum((X_train_nm[nn[:, -1]] - X_test_nm)**2)
# print(dist / len(X_test_nm))
print(dist.shape)
print(np.mean(dist))
y_pred = np.array([np.argmax(np.bincount(y, minlength=10)) for y in y_train[nn]])
# print("k_dist: ", k_dist / len(y_test))
print(np.mean(y_pred == y_test))
