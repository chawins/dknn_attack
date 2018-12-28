import pickle
import numpy as np
import tensorflow as tf
import keras
import falconn
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Flatten, Conv2D, Input, Activation

def load_mnist_model():
    inpt = Input(shape=(28, 28, 1))
    l1 = Conv2D(64, (8, 8), strides=(2, 2), padding='same', activation='relu')(inpt)
    l2 = Conv2D(128, (6, 6), strides=(2, 2), padding='same', activation='relu')(l1)
    l3 = Conv2D(128, (5, 5), strides=(1, 1), padding='valid', activation='relu')(l2)
    flat = Flatten()(l3)
    l4 = Dense(10, activation=None)(flat)
    out = Activation('softmax')(l4)

    model = Model(inputs=inpt, outputs=out)
    l1_rep = Model(inputs=inpt, outputs=l1)
    l2_rep = Model(inputs=inpt, outputs=l2)
    l3_rep = Model(inputs=inpt, outputs=l3)
    l4_rep = Model(inputs=inpt, outputs=l4)

    model.compile(loss=keras.losses.sparse_categorical_crossentropy,
                optimizer=keras.optimizers.Adam(1e-3),
                metrics=['accuracy'])

    # model.fit(X_train, y_train,
    #           batch_size=128,
    #           epochs=15,
    #           verbose=1,
    #           validation_data=(X_test, y_test))

    # model.save_weights('keras_weights/mnist_cnn.h5')

    model.load_weights('keras_weights/mnist_cnn.h5')
    
    return model, [l1_rep, l2_rep, l3_rep, l4_rep]


def compute_cosine(u, v):
    """
    Asssume normalized inputs
    """
    assert u.ndim >= v.ndim
    v_rs = v.reshape(-1)
    if u.ndim > v.ndim:
        u_rs = u.reshape(len(u), -1)
        cdist = np.array([1 - np.dot(u_i, v_rs) for u_i in u_rs])
    else:
        u_rs = u.reshape(-1)
        cdist = 1 - np.dot(u_rs, v_rs)
    return cdist


def find_nn(Q, X, k):
    assert Q.shape[1:] == X.shape[1:]
    nn = np.zeros((len(Q), k), dtype=np.int32)

    axis = tuple(np.arange(1, X.ndim, dtype=np.int32))
    Q_nm = Q / np.sqrt(np.sum(Q**2, axis, keepdims=True))
    X_nm = X / np.sqrt(np.sum(X**2, axis, keepdims=True))

    for i, q in enumerate(Q_nm):
        ind = np.argsort(compute_cosine(X_nm, q))[:k]
        nn[i] = ind
    return nn


def classify(nn, y_X):
    vote = np.array([np.argmax(np.bincount(y)) for y in y_X[nn]])
    return vote


def find_acc(nn, y_Q, y_X):
    vote = classify(nn, y_X)
    acc = np.mean(vote == y_Q)
    print(acc)
    return acc
    
    
def find_nn_diff_class(Q, y_Q, X, y_X, k):
    target = np.zeros((len(Q), k), dtype=np.int32)
    axis = tuple(np.arange(1, X.ndim, dtype=np.int32))
    for i, (q, y_q) in enumerate(zip(Q, y_Q)):
        ind = np.argsort(compute_cosine(X, q))
        target[i] = ind[y_X[ind] != y_q][:k]
    return target


def move_to_target(q, y_q, target, X_nm, y_X, k, n_steps=5):
    """
    Move in straight line to target with binary search.
    Stop when adv is misclassified.
    """
    axis = tuple(np.arange(1, X_nm.ndim, dtype=np.int32))
    line = target - q
    hi = 1
    lo = 0
    adv = None
    for step in range(n_steps):
        mid = (hi + lo)/2
        x_new = mid*line + q
        x_new = x_new / np.sqrt(np.sum(x_new**2))
        # new_neighbors = np.argsort(np.sum((X - x_new)**2, axis=axis))[:k]
        new_neighbors = np.argsort(compute_cosine(X_nm, x_new))[:k]
        y_pred = np.argmax(np.bincount(y_X[new_neighbors]))
        if y_pred == y_q:
            lo = mid
        else:
            hi = mid
            adv = x_new
    return adv


def move_to_target_dknn(q, y_q, target, X, y_X, k, sess, rep, x_ph, query, 
                        n_steps=5):
    """
    Move in straight line to target with binary search.
    Stop when adv is misclassified.
    """
    axis = tuple(np.arange(1, X.ndim, dtype=np.int32))
    line = target - q
    hi = 1
    lo = 0
    adv = None
    for step in range(n_steps):
        mid = (hi + lo)/2
        x_new = mid*line + q
        # x_new = x_new / np.sqrt(np.sum(x_new**2))
        # new_neighbors = np.argsort(np.sum((X - x_new)**2, axis=axis))[:k]
        # new_neighbors = np.argsort(compute_cosine(X, x_new))[:k]

        # rep_adv = get_all_rep_nm(sess, x_new[np.newaxis], rep_ts)
        rep_adv = []
        for rep_l in rep:
            r = sess.run(rep_l, feed_dict={x_ph: x_new[np.newaxis]})
            r = r.reshape(1, -1)
            r = r / np.sqrt(np.sum(r**2, 1, keepdims=True))
            rep_adv.append(r)
        _, alphas = dknn_classify(rep_adv, query, y_X)
        y_pred = np.argmin(alphas[0])
        # p, acc = dknn_acc(A, rep_adv, y_q, query, y_X)
        # y_pred = np.argmax(p[0])

        if y_pred == y_q:
            lo = mid
        else:
            hi = mid
            adv = x_new

    return adv


def setup_lsh(X, num_probes=100):
    assert X.ndim == 2
    params_cp = falconn.LSHConstructionParameters()
    params_cp.dimension = X.shape[1]
    params_cp.lsh_family = falconn.LSHFamily.CrossPolytope
    params_cp.distance_function = falconn.DistanceFunction.EuclideanSquared
    params_cp.l = 100
    params_cp.num_rotations = 1
    params_cp.seed = 1234
    params_cp.num_setup_threads = 0
    params_cp.storage_hash_table = falconn.StorageHashTable.BitPackedFlatHashTable
    falconn.compute_number_of_hash_functions(16, params_cp)
    
    table = falconn.LSHIndex(params_cp)
    table.setup(X)
    query_object = table.construct_query_object()
    query_object.set_num_probes(num_probes)
    
    return query_object


def get_rep_nm(sess, X, rep_ts):
    x = tf.placeholder(tf.float32, (None, 28, 28, 1))
    
    rep = rep_ts(x)
    out = np.zeros((len(X), ) + tuple(rep.shape[1:]), dtype=np.float32)
    for i in range(int(np.ceil(len(X)/1000))):
        out[i*1000 : (i + 1)*1000] = sess.run(
            rep, feed_dict={x: X[i*1000 : (i + 1)*1000]})
    
    out = out.reshape(len(out), -1)
    out = out/np.sqrt(np.sum(out**2, 1, keepdims=True))
    return out


def get_all_rep(sess, X, rep_ts):
    x = tf.placeholder(tf.float32, (None, 28, 28, 1))
    
    [l1_rep, l2_rep, l3_rep, l4_rep] = rep_ts
    l1 = l1_rep(x)
    out_l1 = np.zeros((len(X), ) + tuple(l1.shape[1:]), dtype=np.float32)
    for i in range(int(np.ceil(len(X)/1000))):
        out_l1[i*1000 : (i + 1)*1000] = sess.run(
            l1, feed_dict={x: X[i*1000 : (i + 1)*1000]})
        
    l2 = l2_rep(x)
    out_l2 = np.zeros((len(X), ) + tuple(l2.shape[1:]), dtype=np.float32)
    for i in range(int(np.ceil(len(X)/1000))):
        out_l2[i*1000 : (i + 1)*1000] = sess.run(
            l2, feed_dict={x: X[i*1000 : (i + 1)*1000]})
        
    l3 = l3_rep(x)
    out_l3 = np.zeros((len(X), ) + tuple(l3.shape[1:]), dtype=np.float32)
    for i in range(int(np.ceil(len(X)/1000))):
        out_l3[i*1000 : (i + 1)*1000] = sess.run(
            l3, feed_dict={x: X[i*1000 : (i + 1)*1000]})
        
    l4 = l4_rep(x)
    out_l4 = np.zeros((len(X), ) + tuple(l4.shape[1:]), dtype=np.float32)
    for i in range(int(np.ceil(len(X)/1000))):
        out_l4[i*1000 : (i + 1)*1000] = sess.run(
            l4, feed_dict={x: X[i*1000 : (i + 1)*1000]})
        
    return [out_l1, out_l2, out_l3, out_l4]


def get_all_rep_nm(sess, X, rep_ts):
    
    rep = get_all_rep(sess, X, rep_ts)
    rep_nm = [r.reshape(len(r), -1) for r in rep]
    rep_nm = [r/np.sqrt(np.sum(r**2, 1, keepdims=True)) for r in rep_nm]
    return rep_nm


def query_nn(query, X, n_neighbors=75):
    nn = np.zeros((len(X), n_neighbors), dtype=np.int32)
    for i, x in enumerate(X):
        knn = query.find_k_nearest_neighbors(x, n_neighbors)
        nn[i, :len(knn)] = knn
    return nn


def dknn_classify(rep, query, y_train):
    """
    alphas: (n_layers, n_samples, n_label (number of mismatched nn for each class))
    """
    
    alphas = []
    for l in range(4):
        nn = query_nn(query[l], rep[l], 75)
        bincount = [np.bincount(y, minlength=10) for y in y_train[nn]]
        alpha = np.zeros((len(bincount), 10))
        for j, b in enumerate(bincount):
            alpha[j] = np.array([75 - b[label] for label in range(10)])
        alphas.append(alpha)
    alphas = np.array(alphas)
    return alphas, np.sum(alphas, axis=0)


def dknn_acc(A, rep, y_test, query, y_train):
    
    _, sum_alphas = dknn_classify(rep, query, y_train)
    
    p = np.zeros((len(rep[0]), 10))
    sum_A = np.sum(A, 0)
    for i, s in enumerate(sum_alphas):
        p[i] = np.array([np.sum(ss <= sum_A) for ss in s]) / len(sum_A)
        
    acc = np.mean(np.argmax(p, 1) == y_test)
    return p, acc


def find_2nd_nn_l2(Q, y_Q, X, y_X, k):
    assert Q.shape[1:] == X.shape[1:]
    nn = np.zeros((len(Q), k), dtype=np.int32)
    axis = tuple(np.arange(1, X.ndim, dtype=np.int32))
    for i, q in enumerate(Q):
        dist = np.sum((X - q)**2, axis=axis)
        ind = np.argsort(dist)
        mean_dist = np.zeros((10,))
        for j in range(10):
            ind_j = ind[y_X[ind] == j]
            dist_j = dist[ind_j][:k]
            mean_dist[j] = np.mean(dist_j)
        ind_dist = np.argsort(mean_dist)
        if ind_dist[0] == y_Q[i]:
            nn[i] = ind[y_X[ind] == ind_dist[1]][:k]
        else:
            nn[i] = ind[y_X[ind] == ind_dist[0]][:k]
    return nn