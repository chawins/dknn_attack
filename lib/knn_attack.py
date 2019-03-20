import sys
import timeit

import numpy as np
import tensorflow as tf

from lib.lib_knn import *


class AttackV5(object):

    def __init__(self, sess, model, get_rep, X, X_rep, y_X, A, query,
                 pert_norm=np.inf, batch_size=1000, lr=1e-3,
                 abort_early=True, init_const=1, pert_bound=0.3,
                 min_dist=None, m=100, target_nn=38):
        """
        X_rep must be normalized and flattened
        m: (int) number of neighbors to consider
        """

        self.sess = sess
        self.model = model
        self.X = X
        self.X_rep = X_rep
        self.y_X = y_X
        self.A = A
        self.query = query
        self.n_layers = len(X_rep)
        self.batch_size = batch_size
        self.abort_early = abort_early
        self.init_const = init_const
        self.pert_norm = pert_norm
        self.pert_bound = pert_bound
        # If min_dist is not given, assign arbritary number
        if min_dist is None:
            min_dist = [0.1] * self.n_layers
        self.min_dist = min_dist
        self.m = m
        self.target_nn = target_nn
        self.get_rep = get_rep

        assert self.n_layers == len(get_rep)

        input_ndim = X.ndim
        input_axis = np.arange(1, input_ndim)
        input_shape = (batch_size, ) + X.shape[1:]

        # =============== Set up variables and placeholders =============== #
        # Objective variable
        modifier = tf.Variable(np.zeros(input_shape), dtype=tf.float32)

        # These are variables to be more efficient in sending data to tf
        q_var = tf.Variable(np.zeros(input_shape),
                            dtype=tf.float32, name='q_var')
        x_var = []
        for l in range(self.n_layers):
            rep_shape = (batch_size, m, X_rep[l].shape[1])
            x_var.append(tf.Variable(np.zeros(rep_shape),
                                     dtype=tf.float32,
                                     name='x_var_{}'.format(l)))
        w_var = tf.Variable(
            np.zeros((batch_size, m, 1)), dtype=tf.float32, name='w_var')
        const_var = tf.Variable(
            np.zeros(batch_size), dtype=tf.float32, name='const_var')
        steep_var = tf.Variable(
            np.zeros((batch_size, 1, 1)), dtype=tf.float32, name='const_var')
        clipmin_var = tf.Variable(
            np.zeros(input_shape), dtype=tf.float32, name='clipmin_var')
        clipmax_var = tf.Variable(
            np.zeros(input_shape), dtype=tf.float32, name='clipmax_var')

        # and here's what we use to assign them
        self.assign_q = tf.placeholder(
            tf.float32, input_shape, name='assign_q')
        self.assign_x = []
        for l in range(self.n_layers):
            rep_shape = (batch_size, m, X_rep[l].shape[1])
            self.assign_x.append(tf.placeholder(tf.float32,
                                                rep_shape,
                                                name='assign_x_{}'.format(l)))
        self.assign_w = tf.placeholder(
            tf.float32, [batch_size, m, 1], name='assign_w')
        self.assign_const = tf.placeholder(
            tf.float32, [batch_size], name='assign_const')
        self.assign_steep = tf.placeholder(
            tf.float32, [batch_size, 1, 1], name='assign_steep')
        self.assign_clipmin = tf.placeholder(
            tf.float32, input_shape, name='assign_clipmin')
        self.assign_clipmax = tf.placeholder(
            tf.float32, input_shape, name='assign_clipmax')

        # ================= Get reprentation tensor ================= #
        # Clip to ensure pixel value is between 0 and 1
        self.new_q = (tf.tanh(modifier + q_var) + 1) / 2
        self.new_q = self.new_q * (clipmax_var - clipmin_var) + clipmin_var
        # Distance to the input data
        orig = (tf.tanh(q_var) + 1) / \
            2 * (clipmax_var - clipmin_var) + clipmin_var
        self.rep = []
        for l in range(self.n_layers):
            rep = get_rep[l](self.new_q)
            rep = tf.reshape(rep, [batch_size, 1, -1])
            rep = rep / tf.norm(rep, axis=2, keepdims=True)
            self.rep.append(rep)

        # L2 perturbation loss
        l2dist = tf.reduce_sum(tf.square(self.new_q - orig), input_axis)
        self.l2dist = tf.maximum(0., l2dist - self.pert_bound**2)

        # ================== Approximate NN loss ================== #

        def sigmoid(x, a=1):
            return 1 / (1 + tf.exp(-a * x))

        self.nn_loss = 0
        for l in range(self.n_layers):
            dist = tf.norm(self.rep[l] - x_var[l], axis=2, keepdims=True)
            self.nn_loss += tf.reduce_sum(
                w_var * sigmoid(min_dist[l] - dist, steep_var), (1, 2))

        # ==================== Setup optimizer ==================== #
        if pert_norm == 2:
            # For L-2 norm constraint, we use a penalty term
            self.loss = tf.reduce_mean(self.nn_loss + const_var * self.l2dist)
        elif pert_norm == np.inf:
            self.loss = tf.reduce_mean(self.nn_loss)
        else:
            raise ValueError('Invalid choice of perturbation norm!')

        # DEBUG
        self.dist = dist
        self.rep_db = rep
        self.gradient = tf.gradients(self.nn_loss, modifier)
        print('rep: ', self.rep)
        # dist: (batch_size, m, 1)
        print('dist: ', dist)
        # sigmoid: (batch_size, m, 1)
        print('sigmoid: ', sigmoid(min_dist[l] - dist, steep_var))
        # weights: (batch_size, m, 1)
        print('weights: ', w_var * sigmoid(min_dist[l] - dist, steep_var))
        # loss, nn_loss: (batch_size, )
        print('loss: ', tf.reduce_sum(
            w_var * sigmoid(min_dist[l] - dist, steep_var), (1, 2)))
        print('nn_loss: ', self.nn_loss)

        start_vars = set(x.name for x in tf.global_variables())
        optimizer = tf.train.AdamOptimizer(lr)
        self.train_step = optimizer.minimize(self.loss, var_list=[modifier])
        end_vars = tf.global_variables()
        new_vars = [x for x in end_vars if x.name not in start_vars]

        self.setup = []
        self.setup.append(q_var.assign(self.assign_q))
        self.setup.extend([x_var[l].assign(self.assign_x[l])
                           for l in range(self.n_layers)])
        self.setup.append(w_var.assign(self.assign_w))
        self.setup.append(steep_var.assign(self.assign_steep))
        self.setup.append(const_var.assign(self.assign_const))
        self.setup.append(clipmin_var.assign(self.assign_clipmin))
        self.setup.append(clipmax_var.assign(self.assign_clipmax))
        self.init = tf.variables_initializer(var_list=[modifier] + new_vars)

    def attack(self, Q, y_Q, bin_search_steps=5, max_iter=200,
               rnd_start=0):
        r = []
        for i in range(0, len(Q), self.batch_size):
            print("Running Baseline Attack on instance {} of {}".format(
                i, len(Q)))
            t1 = timeit.default_timer()
            Q_batch = Q[i:i + self.batch_size]
            y_batch = y_Q[i:i + self.batch_size]
            real_len = Q_batch.shape[0]
            if real_len != self.batch_size:
                pad_Q = ((0, self.batch_size - real_len),
                         (0, 0), (0, 0), (0, 0))
                pad_y = ((0, self.batch_size - real_len))
                Q_batch = np.pad(Q_batch, pad_Q, 'constant')
                y_batch = np.pad(y_batch, pad_y, 'constant')
            r.extend(self.attack_batch(Q_batch,
                                       y_batch,
                                       bin_search_steps=bin_search_steps,
                                       max_iter=max_iter,
                                       rnd_start=rnd_start))
            t2 = timeit.default_timer()
            print("Time this batch: {:.0f}s".format(t2 - t1))
        return np.array(r)[:len(Q)]

    def attack_batch(self, Q, y_Q, bin_search_steps=5, max_iter=200,
                     rnd_start=0):

        # Find closest rep of different class
        print("  Finding nn representation as target...")

        # TODO:
        # nn = find_nn_diff_class_l2(Q, y_Q, self.X, self.y_X, self.m)
        # nn = find_nn_l2(Q, self.X, self.m)
        # Get 1st-layer rep and find m nearest neighbors of one wrong class
        l = 0
        Q_rep = get_rep_nm(self.sess, Q, self.get_rep[l])
        nn = find_2nd_nn_l2(Q_rep, y_Q, self.X_rep[l], self.y_X, self.m)

        rep_m = [np.squeeze(rep[nn]) for rep in self.X_rep]

        # Get weights w
        w = 2 * (self.y_X[nn] == y_Q[:, np.newaxis]).astype(np.float32) - 1

        # Initialize steep
        steep = np.ones((self.batch_size, 1, 1)) * 4

        # Find nn to target rep to save nn search time during optimization
        # check_rep = find_nn(target_rep, self.X_rep, 100)

        o_bestl2 = np.zeros((self.batch_size, )) + 1e9
        o_bestadv = np.zeros_like(Q[:self.batch_size], dtype=np.float32)

        # Set the lower and upper bounds
        lower_bound = np.zeros(self.batch_size)
        const = np.ones(self.batch_size) * self.init_const
        upper_bound = np.ones(self.batch_size) * 1e9

        if self.pert_norm == np.inf:
            bin_search_steps = 1

        for outer_step in range(bin_search_steps):

            noise = rnd_start * np.random.rand(*Q.shape)
            Q_tanh = np.clip(Q + noise, 0., 1.)

            # Calculate bound with L-inf norm constraints
            if self.pert_norm == np.inf:
                # Re-scale instances to be within range [x-d, x+d]
                # for d is pert_bound
                clipmin = np.clip(Q_tanh - self.pert_bound, 0., 1.)
                clipmax = np.clip(Q_tanh + self.pert_bound, 0., 1.)

            # Calculate bound with L2 norm constraints
            elif self.pert_norm == 2:
                # Re-scale instances to be within range [0, 1]
                clipmin = np.zeros_like(Q_tanh)
                clipmax = np.ones_like(Q_tanh)

            Q_tanh = (Q_tanh - clipmin) / (clipmax - clipmin)
            Q_tanh = (Q_tanh * 2) - 1
            Q_tanh = np.arctanh(Q_tanh * .999999)
            Q_batch = Q_tanh[:self.batch_size]

            bestl2 = np.zeros((self.batch_size, )) + 1e9
            bestadv = np.zeros_like(Q_batch, dtype=np.float32)
            print("  Binary search step {} of {}".format(
                outer_step, bin_search_steps))

            # Set the variables so that we don't have to send them over again
            self.sess.run(self.init)
            setup_dict = {self.assign_q: Q_batch,
                          self.assign_w: w[:, :, np.newaxis],
                          self.assign_const: const,
                          self.assign_steep: steep,
                          self.assign_clipmin: clipmin,
                          self.assign_clipmax: clipmax}
            for l in range(self.n_layers):
                setup_dict[self.assign_x[l]] = rep_m[l]
            self.sess.run(self.setup, feed_dict=setup_dict)

            prev = 1e6
            for iteration in range(max_iter):
                # Take one step in optimization
                _, l, l2s, qs, reps = self.sess.run([self.train_step,
                                                     self.loss,
                                                     self.l2dist,
                                                     self.new_q,
                                                     self.rep])

                # DEBUG
                # print(self.sess.run(self.dist))
                # grad = self.sess.run(self.gradient)
                # print(grad)
                # print(np.max(qs[0]))
                # print(np.min(qs[0]))
                # rep = self.sess.run(self.rep_db)
                # print(rep.shape)
                # print(np.sum(rep**2, axis=2))
                # print(np.sum(qs, (1,2,3)))

                if iteration % (max_iter // 10) == 0:
                    print(("    Iteration {} of {}: loss={:.3g} l2={:.3g}").format(
                        iteration, max_iter, l, np.mean(l2s)))

                # Abort early if stop improving
                if self.abort_early and iteration % (max_iter // 10) == 0:
                    if l > prev * .9999:
                        print("    Failed to make progress; stop early")
                        break
                    prev = l

                # Check success of adversarial examples
                # check_iter = [int(max_iter * 0.8),
                #               int(max_iter * 0.9), int(max_iter - 1)]
                # if iteration in check_iter:
                #     reps = [np.squeeze(rep) for rep in reps]
                #     p, acc = dknn_acc(self.A, reps, y_Q, self.query, self.y_X)
                #     print(acc)
                #     suc_ind = np.where(np.argmax(p, 1) != y_Q)[0]
                #     for ind in suc_ind:
                #         if l2s[ind] < bestl2[ind]:
                #             bestl2[ind] = l2s[ind]
                #             bestadv[ind] = qs[ind]

            # Adjust const according to results
            for e in range(self.batch_size):
                if bestl2[e] < 1e9:
                    # Success, divide const by two
                    upper_bound[e] = min(upper_bound[e], const[e])
                    if upper_bound[e] < 1e9:
                        const[e] = (lower_bound[e] + upper_bound[e]) / 2
                    if bestl2[e] < o_bestl2[e]:
                        o_bestl2[e] = bestl2[e]
                        o_bestadv[e] = bestadv[e]
                else:
                    # Failure, either multiply by 10 if no solution found yet
                    #          or do binary search with the known upper bound
                    lower_bound[e] = max(lower_bound[e], const[e])
                    if upper_bound[e] < 1e9:
                        const[e] = (lower_bound[e] + upper_bound[e]) / 2
                    else:
                        const[e] *= 10

        # Also save unsuccessful samples
        ind = np.where(o_bestl2 == 1e9)[0]
        o_bestadv[ind] = qs[ind]

        return o_bestadv


class BaselineAttack(object):

    def __init__(self, sess, model, get_rep, X, X_rep, y_X, A, query,
                 pert_norm=2, batch_size=1000, lr=1e-3,
                 abort_early=True, init_const=1, min_dist=1,
                 pert_bound=0.3):
        """
        X_rep must be normalized
        """

        self.sess = sess
        self.model = model
        self.get_rep = get_rep
        self.X = X
        self.X_rep = X_rep
        self.y_X = y_X
        self.A = A
        self.query = query
        self.batch_size = batch_size
        self.abort_early = abort_early
        self.init_const = init_const
        self.min_dist = min_dist
        self.pert_norm = pert_norm
        self.pert_bound = pert_bound
        self.n_layers = len(get_rep)

        input_ndim = X.ndim
        input_axis = np.arange(1, input_ndim)
        input_shape = (batch_size, ) + X.shape[1:]
        rep_ndim = X_rep.ndim
        rep_axis = np.arange(1, rep_ndim)
        rep_shape = (batch_size, ) + X_rep.shape[1:]

        # Objective variable
        modifier = tf.Variable(np.zeros(input_shape), dtype=tf.float32)

        # These are variables to be more efficient in sending data to tf
        q_var = tf.Variable(np.zeros(input_shape),
                            dtype=tf.float32, name='q_var')
        target_var = tf.Variable(
            np.zeros(rep_shape), dtype=tf.float32, name='target_var')
        const_var = tf.Variable(
            np.zeros(batch_size), dtype=tf.float32, name='const_var')

        # and here's what we use to assign them
        self.assign_q = tf.placeholder(
            tf.float32, input_shape, name='assign_q')
        self.assign_target = tf.placeholder(
            tf.float32, rep_shape, name='assign_target')
        self.assign_const = tf.placeholder(
            tf.float32, [batch_size], name='assign_const')

        # Clip to ensure pixel value is between 0 and 1
        self.new_q = tf.clip_by_value(q_var + modifier, 0., 1.)
        # Get reprentation tensor
        self.rep = []
        for l in range(self.n_layers):
            rep = get_rep[l](self.new_q)
            rep = tf.reshape(rep, [batch_size, -1])
            rep = rep / tf.norm(rep, axis=1, keepdims=True)
            self.rep.append(rep)
        # rep = get_rep(self.new_q)
        # rep = tf.reshape(rep, [batch_size, -1])
        # self.rep = rep / tf.norm(rep, axis=1, keepdims=True)

        # L2 perturbation loss
        l2dist = tf.reduce_sum(tf.square(modifier), input_axis)
        self.l2dist = tf.maximum(0., l2dist - self.pert_bound**2)
        # Similarity loss
        dist_loss = tf.reduce_sum(
            tf.square(self.rep[0] - target_var), rep_axis)
        self.dist_loss = tf.maximum(0., dist_loss - self.min_dist**2)

        # Setup optimizer
        start_vars = set(x.name for x in tf.global_variables())
        if pert_norm == 2:
            # For L-2 norm constraint, we use Adam optimizer with
            # a penalty term
            self.loss = tf.reduce_mean(
                self.dist_loss + const_var * self.l2dist)
            optimizer = tf.train.AdamOptimizer(lr)
            self.train_step = optimizer.minimize(
                self.loss, var_list=[modifier])
        elif pert_norm == np.inf:
            # For L-inf norm constraint, we use L-BFGS-B optimizer
            # to provide correct bound, optimizer setup is moved to attack()
            self.loss = tf.reduce_mean(self.dist_loss)
            self.modifier = modifier
        else:
            raise ValueError('Invalid choice for perturbation norm!')

        end_vars = tf.global_variables()
        new_vars = [x for x in end_vars if x.name not in start_vars]

        self.setup = []
        self.setup.append(q_var.assign(self.assign_q))
        self.setup.append(target_var.assign(self.assign_target))
        self.setup.append(const_var.assign(self.assign_const))
        self.init = tf.variables_initializer(var_list=[modifier] + new_vars)

    def attack(self, Q, y_Q, bin_search_steps=5, max_iter=200):
        r = []
        for i in range(0, len(Q), self.batch_size):
            print("Running Baseline Attack on instance {} of {}".format(
                i, len(Q)))
            t1 = timeit.default_timer()
            Q_batch = Q[i:i + self.batch_size]
            y_batch = y_Q[i:i + self.batch_size]
            real_len = Q_batch.shape[0]
            if real_len != self.batch_size:
                pad_Q = ((0, self.batch_size - real_len),
                         (0, 0), (0, 0), (0, 0))
                pad_y = ((0, self.batch_size - real_len))
                Q_batch = np.pad(Q_batch, pad_Q, 'constant')
                y_batch = np.pad(y_batch, pad_y, 'constant')
            r.extend(self.attack_batch(Q_batch,
                                       y_batch,
                                       bin_search_steps=bin_search_steps,
                                       max_iter=max_iter))
            t2 = timeit.default_timer()
            print("Time this batch: {:.0f}s".format(t2 - t1))
        return np.array(r)[:len(Q)]

    def attack_batch(self, Q, y_Q, bin_search_steps=5, max_iter=200):

        # Find closest rep of different class
        print("  Finding nn representation as target...")
        Q_rep = get_rep_nm(self.sess, Q, self.get_rep[0])
        nn = find_nn_diff_class(Q_rep, y_Q, self.X_rep, self.y_X, 1)
        target_rep = np.squeeze(self.X_rep[nn])
        # Find nn to target rep to save nn search time during optimization
        # check_rep = find_nn(target_rep, self.X_rep, 100)

        # ============ Optimizing with L-inf norm constraints =========== #
        # L-BFGS-B optimizer only needs to be called once
        if self.pert_norm == np.inf:
            self.sess.run(self.init)
            Q_batch = Q[:self.batch_size]
            target_rep_batch = target_rep[:self.batch_size]
            const = np.ones(self.batch_size) * self.init_const

            # Set the variables so that we don't have to send them over again
            self.sess.run(
                self.setup, {
                    self.assign_q: Q_batch,
                    self.assign_target: target_rep_batch,
                    self.assign_const: const
                })

            # Set up variables bound and optimizer
            upper_bound = np.minimum(self.pert_bound, 1 - Q_batch)
            lower_bound = np.maximum(-self.pert_bound, -Q_batch)
            var_to_bounds = {self.modifier: (lower_bound, upper_bound)}
            optimizer = tf.contrib.opt.ScipyOptimizerInterface(
                self.loss,
                var_list=[self.modifier],
                var_to_bounds=var_to_bounds,
                method='L-BFGS-B')

            # Call optimizer
            optimizer.minimize(self.sess)
            return self.sess.run(self.new_q)

        # ============= Optimizing with L2 norm constraints ============ #
        o_bestl2 = np.zeros((self.batch_size, )) + 1e9
        o_bestadv = np.zeros_like(Q[:self.batch_size])

        # Set the lower and upper bounds
        lower_bound = np.zeros(self.batch_size)
        const = np.ones(self.batch_size) * self.init_const
        upper_bound = np.ones(self.batch_size) * 1e9

        for outer_step in range(bin_search_steps):

            self.sess.run(self.init)
            Q_batch = Q[:self.batch_size]
            target_rep_batch = target_rep[:self.batch_size]

            bestl2 = np.zeros((self.batch_size, )) + 1e9
            bestadv = np.zeros_like(Q_batch)
            print("  Binary search step {} of {}".format(
                outer_step, bin_search_steps))

            # Set the variables so that we don't have to send them over again
            self.sess.run(
                self.setup, {
                    self.assign_q: Q_batch,
                    self.assign_target: target_rep_batch,
                    self.assign_const: const
                })

            prev = 1e6
            for iteration in range(max_iter):
                # Take one step in optimization
                _, l, l2s, dls, qs, reps = self.sess.run([self.train_step,
                                                          self.loss,
                                                          self.l2dist,
                                                          self.dist_loss,
                                                          self.new_q,
                                                          self.rep])

                if iteration % (max_iter // 10) == 0:
                    print(("    Iteration {} of {}: loss={:.3g} l2={:.3g}").format(
                        iteration, max_iter, l, np.mean(l2s)))

                # Abort early if stop improving
                if self.abort_early and iteration % (max_iter // 10) == 0:
                    if l > prev * .9999:
                        reps = [np.squeeze(rep) for rep in reps]
                        p, acc = dknn_acc(self.A, reps, y_Q,
                                          self.query, self.y_X)
                        print(acc)
                        suc_ind = np.where(np.argmax(p, 1) != y_Q)[0]
                        for ind in suc_ind:
                            if l2s[ind] < bestl2[ind]:
                                bestl2[ind] = l2s[ind]
                                bestadv[ind] = qs[ind]
                        print("    Failed to make progress; stop early")
                        break
                    prev = l

                # Check termination condition
                # if iteration % (max_iter // 10) == 0:
                #     suc_ind = np.where(l2s < 1e-3)[0]
                #     for ind in suc_ind:
                #         if l2s[ind] < bestl2[ind]:
                #             bestl2[ind] = l2s[ind]
                #             bestadv[ind] = qs[ind]

                # check_iter = [int(max_iter * 0.8),
                #               int(max_iter * 0.9), int(max_iter - 1)]
                # if iteration in check_iter:
                #     qs, reps = self.sess.run([self.new_q, self.rep])
                #     reps = [np.squeeze(rep) for rep in reps]
                #     p, acc = dknn_acc(self.A, reps, y_Q, self.query, self.y_X)
                #     print(acc)
                #     suc_ind = np.where(np.argmax(p, 1) != y_Q)[0]
                #     for ind in suc_ind:
                #         if l2s[ind] < bestl2[ind]:
                #             bestl2[ind] = l2s[ind]
                #             bestadv[ind] = qs[ind]

            # Adjust const according to results
            for e in range(self.batch_size):
                if bestl2[e] < 1e9:
                    # Success, divide const by two
                    upper_bound[e] = min(upper_bound[e], const[e])
                    if upper_bound[e] < 1e9:
                        const[e] = (lower_bound[e] + upper_bound[e]) / 2
                    if bestl2[e] < o_bestl2[e]:
                        o_bestl2[e] = bestl2[e]
                        o_bestadv[e] = bestadv[e]
                else:
                    # Failure, either multiply by 10 if no solution found yet
                    #          or do binary search with the known upper bound
                    lower_bound[e] = max(lower_bound[e], const[e])
                    if upper_bound[e] < 1e9:
                        const[e] = (lower_bound[e] + upper_bound[e]) / 2
                    else:
                        const[e] *= 10

        # Also save unsuccessful samples
        ind = np.where(o_bestl2 == 1e9)[0]
        o_bestadv[ind] = qs[ind]

        return o_bestadv


# def naive_attack(Q, y_Q, X, y_X, k, n_steps=5):
#     """
#     Naive attack (untargeted):
#     Complexity is O(k * n_X log (n_X) * n_Q * n_steps)

#     1. Choose trianing sample of target class from X closest to query
#     2. Find closest sample of the same class to the mean of K
#     3. Add that sample to K
#     4. Repeat 2. - 3. until |K| = k/2 + 1
#     5. Move query closer to mean of K, terminate when query becomes
#        adversarial
#     """

#     nn = find_nn_diff_class(Q, y_Q, X, y_X, 1)
#     X_adv = np.zeros_like(Q)
#     axis = tuple(np.arange(1, X.ndim, dtype=np.int32))

#     for i, (q, y_q) in enumerate(zip(Q, y_Q)):

#         if i % 200 == 0:
#             print(i)

#         n_neighbors = int(np.ceil(k/2))
#         K = np.zeros((n_neighbors, ) + Q.shape[1:])

#         # Step 1.
#         K[0] = X[nn[i, 0]]
#         y_adv = y_X[nn[i, 0]]
#         K_ind = [nn[i, 0]]

#         for j in range(1, n_neighbors):

#             # Step 2.
#             mean = np.mean(K[:j], axis=0)
#             mean = mean / np.sqrt(np.sum(mean**2))
#             ind = np.argsort(compute_cosine(X, mean))
#             # ind = np.argsort(np.sum((X - mean)**2, axis=axis))
#             new_nbs = ind[y_X[ind] != y_adv]

#             # Step 3.
#             for new_nb in new_nbs:
#                 if new_nb not in K_ind:
#                     K_ind.append(new_nb)
#                     K[j] = X[new_nb]
#                     break

#         # Step 5.
#         mean = np.mean(K, axis=0)
#         mean = mean / np.sqrt(np.sum(mean**2))
#         X_adv[i] = move_to_target(q, y_q, mean, X, y_X, k, n_steps)

#     return X_adv

def naive_attack(Q, y_Q, X, y_X, k, n_steps=5):
    """
    Naive attack (untargeted):
    Complexity is O(k * n_X log (n_X) * n_Q * n_steps)

    1. Choose trianing sample of target class from X closest to query
    2. Find closest sample of the same class to the mean of K
    3. Add that sample to K
    4. Repeat 2. - 3. until |K| = k/2 + 1
    5. Move query closer to mean of K, terminate when query becomes
       adversarial
    """

    # nn = find_nn_diff_class(Q, y_Q, X, y_X, 1)

    nn = np.zeros((len(Q), ), dtype=np.int32)
    axis = tuple(np.arange(1, X.ndim, dtype=np.int32))
    X_adv = np.zeros_like(Q)
    X_nm = X / np.sqrt(np.sum(X**2, axis=axis, keepdims=True))

    for i, (q, y_q) in enumerate(zip(Q, y_Q)):

        if i % 200 == 0:
            print(i)

        t1 = timeit.default_timer()

        n_neighbors = int(np.ceil(k / 2))
        K = np.zeros((n_neighbors, ) + Q.shape[1:])
        ind = np.argsort(np.sum((X - q)**2, axis=axis))
        nn = ind[y_X[ind] != y_q][0]

        # Step 1.
        K[0] = X[nn]
        y_adv = y_X[nn]
        K_ind = [nn]

        for j in range(1, n_neighbors):

            # Step 2.
            mean = np.mean(K[:j], axis=0)
            # mean = mean / np.sqrt(np.sum(mean**2))
            # ind = np.argsort(compute_cosine(X, mean))
            ind = np.argsort(np.sum((X - mean)**2, axis=axis))
            new_nbs = ind[y_X[ind] != y_adv]

            # Step 3.
            for new_nb in new_nbs:
                if new_nb not in K_ind:
                    K_ind.append(new_nb)
                    K[j] = X[new_nb]
                    break

        # Step 5.
        mean = np.mean(K, axis=0)
        # mean = mean / np.sqrt(np.sum(mean**2))
        X_adv[i] = move_to_target(q, y_q, mean, X_nm, y_X, k, n_steps)

        t2 = timeit.default_timer()
        print(t2 - t1)

    return X_adv


class KNNAttack(object):

    def __init__(self, sess, X, y_X,
                 pert_norm=2, batch_size=1000, lr=1e-3,
                 abort_early=True, init_const=1, min_dist=1,
                 pert_bound=0.3, m=100):
        """
        """

        self.sess = sess
        self.X = X
        self.y_X = y_X
        self.batch_size = batch_size
        self.abort_early = abort_early
        self.init_const = init_const
        self.min_dist = min_dist
        self.pert_norm = pert_norm
        self.pert_bound = pert_bound
        self.m = m

        input_ndim = X.ndim
        input_axis = np.arange(1, input_ndim)
        input_shape = (batch_size, ) + X.shape[1:]

        # Objective variable
        modifier = tf.Variable(np.zeros(input_shape), dtype=tf.float32)

        # These are variables to be more efficient in sending data to tf
        q_var = tf.Variable(np.zeros(input_shape),
                            dtype=tf.float32, name='q_var')
        x_var = tf.Variable(np.zeros((batch_size, m) + X.shape[1:]),
                            dtype=tf.float32,
                            name='x_var')
        const_var = tf.Variable(
            np.zeros(batch_size), dtype=tf.float32, name='const_var')
        w_var = tf.Variable(
            np.zeros((batch_size, m, 1)), dtype=tf.float32, name='w_var')
        steep_var = tf.Variable(
            np.zeros((batch_size, 1, 1)), dtype=tf.float32, name='const_var')
        clipmin_var = tf.Variable(
            np.zeros(input_shape), dtype=tf.float32, name='clipmin_var')
        clipmax_var = tf.Variable(
            np.zeros(input_shape), dtype=tf.float32, name='clipmax_var')

        # and here's what we use to assign them
        self.assign_q = tf.placeholder(
            tf.float32, input_shape, name='assign_q')
        self.assign_x = tf.placeholder(
            tf.float32, (batch_size, m) + X.shape[1:], name='assign_x')
        self.assign_const = tf.placeholder(
            tf.float32, [batch_size], name='assign_const')
        self.assign_w = tf.placeholder(
            tf.float32, [batch_size, m, 1], name='assign_w')
        self.assign_steep = tf.placeholder(
            tf.float32, [batch_size, 1, 1], name='assign_steep')
        self.assign_clipmin = tf.placeholder(
            tf.float32, input_shape, name='assign_clipmin')
        self.assign_clipmax = tf.placeholder(
            tf.float32, input_shape, name='assign_clipmax')

        # Change of variables
        self.new_q = (tf.tanh(modifier + q_var) + 1) / 2
        self.new_q = self.new_q * (clipmax_var - clipmin_var) + clipmin_var
        # Distance to the input data
        orig = (tf.tanh(q_var) + 1) / \
            2 * (clipmax_var - clipmin_var) + clipmin_var
        # L2 perturbation loss
        l2dist = tf.reduce_sum(tf.square(self.new_q - orig), input_axis)
        self.l2dist = tf.maximum(0., l2dist - self.pert_bound**2)

        def sigmoid(x, a=1):
            return 1 / (1 + tf.exp(-a * x))

        self.new_q_rs = tf.reshape(self.new_q, (batch_size, 1, -1))
        self.new_q_rs = self.new_q_rs / \
            tf.norm(self.new_q_rs, axis=2, keepdims=True)
        x_var_rs = tf.reshape(x_var, (batch_size, m, -1))
        dist = tf.norm(self.new_q_rs - x_var_rs, axis=2, keepdims=True)
        self.nn_loss = tf.reduce_sum(
            w_var * sigmoid(min_dist - dist, steep_var), (1, 2))

        # Setup optimizer
        if pert_norm == 2:
            # For L-2 norm constraint, we use a penalty term
            self.loss = tf.reduce_mean(const_var * self.nn_loss + self.l2dist)
        elif pert_norm == np.inf:
            self.loss = tf.reduce_mean(self.nn_loss)
        else:
            raise ValueError('Invalid choice of perturbation norm!')

        start_vars = set(x.name for x in tf.global_variables())
        optimizer = tf.train.AdamOptimizer(lr)
        self.train_step = optimizer.minimize(self.loss, var_list=[modifier])
        end_vars = tf.global_variables()
        new_vars = [x for x in end_vars if x.name not in start_vars]

        self.setup = []
        self.setup.append(q_var.assign(self.assign_q))
        self.setup.append(const_var.assign(self.assign_const))
        self.setup.append(x_var.assign(self.assign_x))
        self.setup.append(w_var.assign(self.assign_w))
        self.setup.append(steep_var.assign(self.assign_steep))
        self.setup.append(clipmin_var.assign(self.assign_clipmin))
        self.setup.append(clipmax_var.assign(self.assign_clipmax))
        self.init = tf.variables_initializer(var_list=[modifier] + new_vars)

    def attack(self, Q, y_Q, bin_search_steps=5, max_iter=200, rnd_start=0):
        r = []
        for i in range(0, len(Q), self.batch_size):
            print("Running Baseline Attack on instance {} of {}".format(
                i, len(Q)))
            t1 = timeit.default_timer()
            Q_batch = Q[i:i + self.batch_size]
            y_batch = y_Q[i:i + self.batch_size]
            real_len = Q_batch.shape[0]
            if real_len != self.batch_size:
                pad_Q = ((0, self.batch_size - real_len), (0, 0))
                pad_y = ((0, self.batch_size - real_len))
                Q_batch = np.pad(Q_batch, pad_Q, 'constant')
                y_batch = np.pad(y_batch, pad_y, 'constant')
            r.extend(self.attack_batch(Q_batch,
                                       y_batch,
                                       bin_search_steps=bin_search_steps,
                                       max_iter=max_iter,
                                       rnd_start=rnd_start))
            t2 = timeit.default_timer()
            print("Time this batch: {:.0f}s".format(t2 - t1))
        return np.array(r)[:len(Q)]

    def attack_batch(self, Q, y_Q, bin_search_steps=5, max_iter=200, rnd_start=0):

        print("  Finding nn as target...")
        norm = np.sqrt(np.sum(Q**2, axis=1, keepdims=True))
        ind = np.squeeze(norm) != 0
        Q_nm = np.zeros_like(Q)
        Q_nm[ind] = Q[ind] / norm[ind]
        nn = find_2nd_nn_l2(Q_nm, y_Q, self.X, self.y_X, self.m)
        target = self.X[nn]

        # Get weights w
        w = 2 * (self.y_X[nn] == y_Q[:, np.newaxis]).astype(np.float32) - 1

        # Initialize steep
        steep = np.ones((self.batch_size, 1, 1)) * 4

        o_bestl2 = np.zeros((self.batch_size, )) + 1e9
        o_bestadv = np.zeros_like(Q[:self.batch_size], dtype=np.float32)

        # Set the lower and upper bounds
        lower_bound = np.zeros(self.batch_size)
        const = np.ones(self.batch_size) * self.init_const
        upper_bound = np.ones(self.batch_size) * 1e9

        if self.pert_norm == np.inf:
            bin_search_steps = 1

        for outer_step in range(bin_search_steps):

            noise = rnd_start * np.random.rand(*Q.shape)
            Q_tanh = np.clip(Q + noise, 0., 1.)

            # Calculate bound with L-inf norm constraints
            if self.pert_norm == np.inf:
                # Re-scale instances to be within range [x-d, x+d]
                # for d is pert_bound
                clipmin = np.clip(Q_tanh - self.pert_bound, 0., 1.)
                clipmax = np.clip(Q_tanh + self.pert_bound, 0., 1.)

            # Calculate bound with L2 norm constraints
            elif self.pert_norm == 2:
                # Re-scale instances to be within range [0, 1]
                clipmin = np.zeros_like(Q_tanh)
                clipmax = np.ones_like(Q_tanh)

            Q_tanh = (Q_tanh - clipmin) / (clipmax - clipmin)
            Q_tanh = (Q_tanh * 2) - 1
            Q_tanh = np.arctanh(Q_tanh * .999999)
            Q_batch = Q_tanh[:self.batch_size]

            bestl2 = np.zeros((self.batch_size, )) + 1e9
            bestadv = np.zeros_like(Q_batch, dtype=np.float32)
            print("  Binary search step {} of {}".format(
                outer_step, bin_search_steps))

            # Set the variables so that we don't have to send them over again
            self.sess.run(self.init)
            setup_dict = {self.assign_q: Q_batch,
                          self.assign_w: w[:, :, np.newaxis],
                          self.assign_x: target,
                          self.assign_const: const,
                          self.assign_steep: steep,
                          self.assign_clipmin: clipmin,
                          self.assign_clipmax: clipmax}
            self.sess.run(self.setup, feed_dict=setup_dict)

            prev = 1e6
            for iteration in range(max_iter):
                # Take one step in optimization
                _, l, l2s, qs, qs_nm = self.sess.run([self.train_step,
                                                      self.loss,
                                                      self.l2dist,
                                                      self.new_q,
                                                      self.new_q_rs])

                if iteration % (max_iter // 10) == 0:
                    print(("    Iteration {} of {}: loss={:.3g} l2={:.3g}").format(
                        iteration, max_iter, l, np.mean(l2s)))

                # Abort early if stop improving
                if self.abort_early and iteration % (max_iter // 10) == 0:
                    if l > prev * .9999:
                        print("    Failed to make progress; stop early")
                        break
                    prev = l

                # Check success of adversarial examples
                check_iter = [int(max_iter * 0.8),
                              int(max_iter * 0.9), int(max_iter - 1)]
                if iteration in check_iter:
                    nn = find_nn(np.squeeze(qs_nm), self.X, self.m)
                    y_knn = classify(nn, self.y_X)
                    suc_ind = np.where(y_knn != y_Q)[0]
                    print(1 - (len(suc_ind) / self.batch_size))
                    for ind in suc_ind:
                        if l2s[ind] < bestl2[ind]:
                            bestl2[ind] = l2s[ind]
                            bestadv[ind] = qs[ind]

            # Adjust const according to results
            for e in range(self.batch_size):
                if bestl2[e] < 1e9:
                    # Success, divide const by two
                    upper_bound[e] = min(upper_bound[e], const[e])
                    if upper_bound[e] < 1e9:
                        const[e] = (lower_bound[e] + upper_bound[e]) / 2
                    if bestl2[e] < o_bestl2[e]:
                        o_bestl2[e] = bestl2[e]
                        o_bestadv[e] = bestadv[e]
                else:
                    # Failure, either multiply by 10 if no solution found yet
                    #          or do binary search with the known upper bound
                    lower_bound[e] = max(lower_bound[e], const[e])
                    if upper_bound[e] < 1e9:
                        const[e] = (lower_bound[e] + upper_bound[e]) / 2
                    else:
                        const[e] *= 10

        # Also save unsuccessful samples
        # ind = np.where(o_bestl2 == 1e9)[0]
        # o_bestadv[ind] = qs[ind]

        return o_bestadv


def mean_attack(Q, y_Q, X, y_X, k, n_steps=5, sess=None, rep_ts=None,
                query=None, A=None):
    """
    Assume Q, X are normalized
    """

    assert Q.shape[1:] == X.shape[1:]

    X_nm = X / np.sqrt(np.sum(X**2, axis=(1, 2, 3), keepdims=True))
    means = np.zeros((10, ) + Q.shape[1:])
    for i in range(10):
        means[i] = np.mean(X[y_X == i], axis=0)
    # means = means / np.sqrt(np.sum(means**2, axis=(1, 2, 3)))

    if rep_ts is not None:
        x_ph = tf.placeholder(tf.float32, (None, 28, 28, 1))
        rep = []
        for r in rep_ts:
            rep.append(r(x_ph))

    X_adv = np.zeros_like(Q)
    for i, (q, y_q) in enumerate(zip(Q, y_Q)):

        if i % 200 == 0:
            print(i)
            sys.stdout.flush()

        t1 = timeit.default_timer()
        # Find the closest mean
        # dist = compute_cosine(means[np.arange(10) != y_q], q)
        dist = np.sum((means[np.arange(10) != y_q] - q)**2, axis=(1, 2, 3))
        mean = means[np.arange(10) != y_q][np.argmin(dist)]

        # Move to mean
        if sess is None:
            X_adv[i] = move_to_target(q, y_q, mean, X_nm, y_X, k, n_steps)
        else:
            X_adv[i] = move_to_target_dknn(
                q, y_q, mean, X, y_X, k, sess, rep, x_ph, query, n_steps)

        t2 = timeit.default_timer()
        print(t2 - t1)
    return X_adv


class AttackCred(object):

    def __init__(self, sess, model, get_rep, X, X_rep, y_X, A, query,
                 pert_norm=np.inf, batch_size=1000, lr=1e-3,
                 abort_early=True, init_const=1, pert_bound=0.3,
                 min_dist=None, m=100, target_nn=38):
        """
        X_rep must be normalized and flattened
        m: (int) number of neighbors to consider
        """

        self.sess = sess
        self.model = model
        self.X = X
        self.X_rep = X_rep
        self.y_X = y_X
        self.A = A
        self.query = query
        self.n_layers = len(X_rep)
        self.batch_size = batch_size
        self.abort_early = abort_early
        self.init_const = init_const
        self.pert_norm = pert_norm
        self.pert_bound = pert_bound
        # If min_dist is not given, assign arbritary number
        if min_dist is None:
            min_dist = [0.1] * self.n_layers
        self.min_dist = min_dist
        self.m = m
        self.target_nn = target_nn
        self.get_rep = get_rep

        assert self.n_layers == len(get_rep)

        input_ndim = X.ndim
        input_axis = np.arange(1, input_ndim)
        input_shape = (batch_size, ) + X.shape[1:]

        # =============== Set up variables and placeholders =============== #
        # Objective variable
        modifier = tf.Variable(np.zeros(input_shape), dtype=tf.float32)

        # These are variables to be more efficient in sending data to tf
        q_var = tf.Variable(np.zeros(input_shape),
                            dtype=tf.float32, name='q_var')
        x_var = []
        for l in range(self.n_layers):
            rep_shape = (batch_size, m, X_rep[l].shape[1])
            x_var.append(tf.Variable(np.zeros(rep_shape),
                                     dtype=tf.float32,
                                     name='x_var_{}'.format(l)))
        w_var = tf.Variable(
            np.zeros((batch_size, m, 1)), dtype=tf.float32, name='w_var')
        const_var = tf.Variable(
            np.zeros(batch_size), dtype=tf.float32, name='const_var')
        steep_var = tf.Variable(
            np.zeros((batch_size, 1, 1)), dtype=tf.float32, name='const_var')
        clipmin_var = tf.Variable(
            np.zeros(input_shape), dtype=tf.float32, name='clipmin_var')
        clipmax_var = tf.Variable(
            np.zeros(input_shape), dtype=tf.float32, name='clipmax_var')

        # and here's what we use to assign them
        self.assign_q = tf.placeholder(
            tf.float32, input_shape, name='assign_q')
        self.assign_x = []
        for l in range(self.n_layers):
            rep_shape = (batch_size, m, X_rep[l].shape[1])
            self.assign_x.append(tf.placeholder(tf.float32,
                                                rep_shape,
                                                name='assign_x_{}'.format(l)))
        self.assign_w = tf.placeholder(
            tf.float32, [batch_size, m, 1], name='assign_w')
        self.assign_const = tf.placeholder(
            tf.float32, [batch_size], name='assign_const')
        self.assign_steep = tf.placeholder(
            tf.float32, [batch_size, 1, 1], name='assign_steep')
        self.assign_clipmin = tf.placeholder(
            tf.float32, input_shape, name='assign_clipmin')
        self.assign_clipmax = tf.placeholder(
            tf.float32, input_shape, name='assign_clipmax')

        # ================= Get reprentation tensor ================= #
        # Clip to ensure pixel value is between 0 and 1
        self.new_q = (tf.tanh(modifier + q_var) + 1) / 2
        self.new_q = self.new_q * (clipmax_var - clipmin_var) + clipmin_var
        # Distance to the input data
        orig = (tf.tanh(q_var) + 1) / \
            2 * (clipmax_var - clipmin_var) + clipmin_var
        self.rep = []
        for l in range(self.n_layers):
            rep = get_rep[l](self.new_q)
            rep = tf.reshape(rep, [batch_size, 1, -1])
            rep = rep / tf.norm(rep, axis=2, keepdims=True)
            self.rep.append(rep)

        # L2 perturbation loss
        l2dist = tf.reduce_sum(tf.square(self.new_q - orig), input_axis)
        self.l2dist = tf.maximum(0., l2dist - self.pert_bound**2)

        # ================== Approximate NN loss ================== #

        def sigmoid(x, a=1):
            return 1 / (1 + tf.exp(-a * x))

        self.nn_loss = 0
        for l in range(self.n_layers):
            dist = tf.norm(self.rep[l] - x_var[l], axis=2, keepdims=True)
            self.nn_loss += tf.reduce_sum(
                w_var * sigmoid(min_dist[l] - dist, steep_var), (1, 2))

        # ==================== Setup optimizer ==================== #
        if pert_norm == 2:
            # For L-2 norm constraint, we use a penalty term
            self.loss = tf.reduce_mean(const_var * self.nn_loss + self.l2dist)
        elif pert_norm == np.inf:
            self.loss = tf.reduce_mean(self.nn_loss)
        else:
            raise ValueError('Invalid choice of perturbation norm!')

        # DEBUG
        self.dist = dist
        self.rep_db = rep
        self.gradient = tf.gradients(self.nn_loss, modifier)
        print('rep: ', self.rep)
        # dist: (batch_size, m, 1)
        print('dist: ', dist)
        # sigmoid: (batch_size, m, 1)
        print('sigmoid: ', sigmoid(min_dist[l] - dist, steep_var))
        # weights: (batch_size, m, 1)
        print('weights: ', w_var * sigmoid(min_dist[l] - dist, steep_var))
        # loss, nn_loss: (batch_size, )
        print('loss: ', tf.reduce_sum(
            w_var * sigmoid(min_dist[l] - dist, steep_var), (1, 2)))
        print('nn_loss: ', self.nn_loss)

        start_vars = set(x.name for x in tf.global_variables())
        optimizer = tf.train.AdamOptimizer(lr)
        self.train_step = optimizer.minimize(self.loss, var_list=[modifier])
        end_vars = tf.global_variables()
        new_vars = [x for x in end_vars if x.name not in start_vars]

        self.setup = []
        self.setup.append(q_var.assign(self.assign_q))
        self.setup.extend([x_var[l].assign(self.assign_x[l])
                           for l in range(self.n_layers)])
        self.setup.append(w_var.assign(self.assign_w))
        self.setup.append(steep_var.assign(self.assign_steep))
        self.setup.append(const_var.assign(self.assign_const))
        self.setup.append(clipmin_var.assign(self.assign_clipmin))
        self.setup.append(clipmax_var.assign(self.assign_clipmax))
        self.init = tf.variables_initializer(var_list=[modifier] + new_vars)

    def attack(self, Q, y_Q, bin_search_steps=5, max_iter=200,
               rnd_start=0):
        r = []
        for i in range(0, len(Q), self.batch_size):
            print("Running Baseline Attack on instance {} of {}".format(
                i, len(Q)))
            t1 = timeit.default_timer()
            Q_batch = Q[i:i + self.batch_size]
            y_batch = y_Q[i:i + self.batch_size]
            real_len = Q_batch.shape[0]
            if real_len != self.batch_size:
                pad_Q = ((0, self.batch_size - real_len),
                         (0, 0), (0, 0), (0, 0))
                pad_y = ((0, self.batch_size - real_len))
                Q_batch = np.pad(Q_batch, pad_Q, 'constant')
                y_batch = np.pad(y_batch, pad_y, 'constant')
            r.extend(self.attack_batch(Q_batch,
                                       y_batch,
                                       bin_search_steps=bin_search_steps,
                                       max_iter=max_iter,
                                       rnd_start=rnd_start))
            t2 = timeit.default_timer()
            print("Time this batch: {:.0f}s".format(t2 - t1))
        return np.array(r)[:len(Q)]

    def attack_batch(self, Q, y_Q, bin_search_steps=5, max_iter=200,
                     rnd_start=0):

        # Find closest rep of different class
        print("  Finding nn representation as target...")

        # TODO:
        # nn = find_nn_diff_class_l2(Q, y_Q, self.X, self.y_X, self.m)
        # nn = find_nn_l2(Q, self.X, self.m)
        # Get 1st-layer rep and find m nearest neighbors of one wrong class
        l = 0
        Q_rep = get_rep_nm(self.sess, Q, self.get_rep[l])
        nn = find_2nd_nn_l2(Q_rep, y_Q, self.X_rep[l], self.y_X, self.m)

        rep_m = [np.squeeze(rep[nn]) for rep in self.X_rep]

        # Get weights w
        w = 2 * (self.y_X[nn] == y_Q[:, np.newaxis]).astype(np.float32) - 1

        # Initialize steep
        steep = np.ones((self.batch_size, 1, 1)) * 4

        # Find nn to target rep to save nn search time during optimization
        # check_rep = find_nn(target_rep, self.X_rep, 100)

        o_bestl2 = np.zeros((self.batch_size, )) + 1e9
        o_bestadv = np.zeros_like(Q[:self.batch_size], dtype=np.float32)

        # Set the lower and upper bounds
        lower_bound = np.zeros(self.batch_size)
        const = np.ones(self.batch_size) * self.init_const
        upper_bound = np.ones(self.batch_size) * 1e9

        if self.pert_norm == np.inf:
            bin_search_steps = 1

        for outer_step in range(bin_search_steps):

            noise = rnd_start * np.random.rand(*Q.shape)
            Q_tanh = np.clip(Q + noise, 0., 1.)

            # Calculate bound with L-inf norm constraints
            if self.pert_norm == np.inf:
                # Re-scale instances to be within range [x-d, x+d]
                # for d is pert_bound
                clipmin = np.clip(Q_tanh - self.pert_bound, 0., 1.)
                clipmax = np.clip(Q_tanh + self.pert_bound, 0., 1.)

            # Calculate bound with L2 norm constraints
            elif self.pert_norm == 2:
                # Re-scale instances to be within range [0, 1]
                clipmin = np.zeros_like(Q_tanh)
                clipmax = np.ones_like(Q_tanh)

            Q_tanh = (Q_tanh - clipmin) / (clipmax - clipmin)
            Q_tanh = (Q_tanh * 2) - 1
            Q_tanh = np.arctanh(Q_tanh * .999999)
            Q_batch = Q_tanh[:self.batch_size]

            bestl2 = np.zeros((self.batch_size, )) + 1e9
            bestadv = np.zeros_like(Q_batch, dtype=np.float32)
            print("  Binary search step {} of {}".format(
                outer_step, bin_search_steps))

            # Set the variables so that we don't have to send them over again
            self.sess.run(self.init)
            setup_dict = {self.assign_q: Q_batch,
                          self.assign_w: w[:, :, np.newaxis],
                          self.assign_const: const,
                          self.assign_steep: steep,
                          self.assign_clipmin: clipmin,
                          self.assign_clipmax: clipmax}
            for l in range(self.n_layers):
                setup_dict[self.assign_x[l]] = rep_m[l]
            self.sess.run(self.setup, feed_dict=setup_dict)

            prev = 1e6
            for iteration in range(max_iter):
                # Take one step in optimization
                _, l, l2s, qs, reps = self.sess.run([self.train_step,
                                                     self.loss,
                                                     self.l2dist,
                                                     self.new_q,
                                                     self.rep])

                # DEBUG
                # print(self.sess.run(self.dist))
                # grad = self.sess.run(self.gradient)
                # print(grad)
                # print(np.max(qs[0]))
                # print(np.min(qs[0]))
                # rep = self.sess.run(self.rep_db)
                # print(rep.shape)
                # print(np.sum(rep**2, axis=2))
                # print(np.sum(qs, (1,2,3)))

                if iteration % (max_iter // 10) == 0:
                    print(("    Iteration {} of {}: loss={:.3g} l2={:.3g}").format(
                        iteration, max_iter, l, np.mean(l2s)))

                # Abort early if stop improving
                if self.abort_early and iteration % (max_iter // 10) == 0:
                    if l > prev * .9999:
                        print("    Failed to make progress; stop early")
                        break
                    prev = l

                # Check success of adversarial examples
                if iteration == max_iter - 1:
                    reps = [np.squeeze(rep) for rep in reps]
                    p, acc = dknn_acc(self.A, reps, y_Q, self.query, self.y_X)
                    print(acc)
                    suc_ind = np.where(np.argmax(p, 1) != y_Q)[0]
                    for ind in suc_ind:
                        if l2s[ind] < bestl2[ind]:
                            bestl2[ind] = l2s[ind]
                            bestadv[ind] = qs[ind]

            # Adjust const according to results
            for e in range(self.batch_size):
                if bestl2[e] < 1e9:
                    # Success, divide const by two
                    upper_bound[e] = min(upper_bound[e], const[e])
                    if upper_bound[e] < 1e9:
                        const[e] = (lower_bound[e] + upper_bound[e]) / 2
                    if bestl2[e] < o_bestl2[e]:
                        o_bestl2[e] = bestl2[e]
                        o_bestadv[e] = bestadv[e]
                else:
                    # Failure, either multiply by 10 if no solution found yet
                    #          or do binary search with the known upper bound
                    lower_bound[e] = max(lower_bound[e], const[e])
                    if upper_bound[e] < 1e9:
                        const[e] = (lower_bound[e] + upper_bound[e]) / 2
                    else:
                        const[e] *= 10

        # Also save unsuccessful samples
        ind = np.where(o_bestl2 == 1e9)[0]
        o_bestadv[ind] = qs[ind]

        return o_bestadv
