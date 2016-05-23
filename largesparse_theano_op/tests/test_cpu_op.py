import itertools
import unittest

import numpy as np
import scipy

from theano import function
import theano.tensor as T
from theano import config
import theano.tests.unittest_tools as utt
from theano import shared

from op import COST, COST_GRAD, LargeSparseTargets


class TestLargeSparseTargets(unittest.TestCase):
    def setUp(self):
        self.op = LargeSparseTargets

    @staticmethod
    def generate_data():
        d = 7
        D = 3
        K = 2
        p = scipy.misc.comb(D, K)
        m = 5

        V_mat = np.asarray(np.random.rand(D, d), dtype=config.floatX)
        U_mat = np.eye(d, dtype=config.floatX)
        UinvT_mat = np.eye(d, dtype=config.floatX)
        Q_mat = np.dot( V_mat.T, V_mat )

        H_mat = np.asarray(np.random.rand(m, d), dtype=config.floatX)
        all_perm=np.array((list(itertools.permutations(list(range(D))))))
        a = np.zeros((m, D))
        a[:] = np.arange(D)
        b = all_perm[np.random.randint(0,p,size=m)]
        tmp_Y_i = (a.flatten()[(b+D*np.arange(m)[...,np.newaxis]).flatten()]).reshape(a.shape)
        Y_indexes_mat = np.asarray(tmp_Y_i[:,:K], dtype=np.int32)
        Y_values_mat = np.asarray(np.ones((m, K)), dtype=config.floatX)
        Y = np.zeros((m, D))
        for i in range(K):
            Y[np.arange(m), Y_indexes_mat[:,i].flatten()] = 1.
        return V_mat, U_mat, UinvT_mat, Q_mat, H_mat, Y_indexes_mat, Y_values_mat, Y

    @staticmethod
    def numpy_large_sparse_targets(H_mat, U_mat, V_mat, Y):
        # Compute cost with numpy
        pred = np.dot(H_mat, np.dot(U_mat, V_mat.T))
        return np.sum((Y - pred)**2)

    def test_large_sparse_targets(self):
        V_mat, U_mat, UinvT_mat, Q_mat, H_mat, Y_indexes_mat, Y_values_mat, Y = \
            TestLargeSparseTargets.generate_data()

        V = shared(V_mat)
        U = shared(U_mat)
        UinvT = shared(UinvT_mat)
        Q = shared(Q_mat)

        H = T.matrix()
        Y_indexes = T.imatrix()
        Y_values = T.matrix()

        learning_rate = 0.1

        f = function([H, Y_indexes, Y_values], 
                     self.op(COST)(V, U, UinvT, Q, H, Y_indexes, 
                                          Y_values, learning_rate))

        theano_out = f(H_mat, Y_indexes_mat, Y_values_mat)

        # Compute cost with numpy
        pred = np.dot(H_mat, np.dot(U_mat, V_mat.T))
        np_out = np.sum((Y - pred)**2)

        utt.assert_allclose(theano_out, np_out)

    def test_large_sparse_targets_grad(self):
        V_mat, U_mat, UinvT_mat, Q_mat, H_mat, Y_indexes_mat, Y_values_mat, Y = \
            TestLargeSparseTargets.generate_data()

        V = shared(V_mat)
        U = shared(U_mat)
        UinvT = shared(UinvT_mat)
        Q = shared(Q_mat)

        H = T.matrix()
        Y_indexes = T.imatrix()
        Y_values = T.matrix()

        learning_rate = 0.1

        def f(H):
            LST_grad = self.op(COST)(V, U, UinvT, Q, H,
                Y_indexes_mat, Y_values_mat, learning_rate)

            return LST_grad

        utt.verify_grad(f, [H_mat])

    def test_large_sparse_targets_merge(self):
        V_mat, U_mat, UinvT_mat, Q_mat, H_mat, Y_indexes_mat, Y_values_mat, Y = \
            TestLargeSparseTargets.generate_data()

        V = shared(V_mat)
        U = shared(U_mat)
        UinvT = shared(UinvT_mat)
        Q = shared(Q_mat)

        H = T.matrix()
        Y_indexes = T.imatrix()
        Y_values = T.matrix()

        learning_rate = 0.1

        LST = LargeSparseTargets(COST)(V, U, UinvT, Q, H, Y_indexes, 
                                    Y_values, learning_rate)

        grad = T.grad(LST.sum(), wrt=H)

        f = function([H, Y_indexes, Y_values], [LST, grad])

        assert len(f.maker.fgraph.toposort()) == 1
        assert isinstance(f.maker.fgraph.toposort()[0].op,
                          self.op)
        assert f.maker.fgraph.toposort()[0].op.what_to_output == COST_GRAD
