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

pert_norm = 2
pert_bound = 0
lr = 1e-1
init_const = 1e1
m = 75
fname = "knn_adv_pn{}_pb{}_lr{}_c{}_m{}.p".format(pert_norm, pert_bound, lr, init_const, m)
# fname = "knn_naive_adv.p"

from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train[:, :, :, np.newaxis].astype(np.float32) / 255.
X_test = X_test[:, :, :, np.newaxis].astype(np.float32) / 255.
X_train_nm = X_train/np.sqrt(np.sum(X_train**2, axis=(1, 2, 3), keepdims=True))
X_train_nm = X_train_nm.reshape(-1, 784)
X_test_nm = X_test/np.sqrt(np.sum(X_test**2, axis=(1, 2, 3), keepdims=True))
X_test_nm = X_test_nm.reshape(-1, 784)

# Randomly chosen calibrate set 75 samples from each class
ind_cal = np.zeros((750, ), dtype=np.int32)
for i in range(10):
    ind = np.where(y_test == i)[0]
    np.random.shuffle(ind)
    ind_cal[i*75 : (i + 1)*75] = ind[:75]
ind_test = np.arange(len(X_test), dtype=np.int32)
ind_test = np.setdiff1d(ind_test, ind_cal)

X_test = X_test[ind_test]
y_test = y_test[ind_test]

print("Setting up attack...")
# # min_dist = [0.7484518915597577, 0.742514077896593, 0.5667317128548899, 0.19457440222463473]
min_dist = 0.6146243
attack = KNNAttack(sess, X_train_nm, y_train,
                   pert_norm=pert_norm, 
                   batch_size=100, 
                   lr=lr, 
                   init_const=init_const, 
                   min_dist=min_dist,
                   pert_bound=pert_bound, 
                   m=m)

X_adv = attack.attack(X_test.reshape(-1, 784), y_test)
pickle.dump(X_adv, open(fname, "wb"))

# X_adv = naive_attack(X_test, y_test, X_train, y_train, 75, n_steps=5)
# pickle.dump(X_adv, open(fname, "wb"))

X_adv_nm = X_adv / np.sqrt(np.sum(X_adv**2, axis=1, keepdims=True))
nn = find_nn(X_adv_nm, X_train_nm, 75)
y_knn = classify(nn, y_train)
print(np.mean(y_knn == y_test))
print(y_knn)