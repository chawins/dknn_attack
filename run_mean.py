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
import sys


from lib.lib_knn import *
from lib.knn_attack import mean_attack

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
from keras.backend.tensorflow_backend import set_session
set_session(sess)

np.random.seed(1234)

pert_norm = np.inf
pert_bound = 0.15
lr = 5e-1
init_const = 1
m = 75
max_iter = 400
fname = "adv_pn{}_pb{}_lr{}_c{}_m{}.p".format(pert_norm, pert_bound, lr, init_const, m)

from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train[:, :, :, np.newaxis].astype(np.float32) / 255.
X_test = X_test[:, :, :, np.newaxis].astype(np.float32) / 255.

print("Loading model...")
model, rep_ts = load_mnist_model()

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

print("Getting representations...")
rep_train_nm = get_all_rep_nm(sess, X_train, rep_ts)

print("Setting up LSH...")
query = []
for rep in rep_train_nm:
    query.append(setup_lsh(rep, 100))
A = pickle.load(open("A_cosine_lsh.p", "rb"))

X_dknn = mean_attack(X_test, y_test, X_train, y_train, 75, n_steps=5, 
                     sess=sess, rep_ts=rep_ts, query=query, A=A)
pickle.dump(X_dknn, open("dknn_mean.p", "wb"))

# X_knn = mean_attack(X_test, y_test, X_train, y_train, 75, n_steps=5)
# pickle.dump(X_knn, open("knn_mean_pb{}".format(eps), "wb"))