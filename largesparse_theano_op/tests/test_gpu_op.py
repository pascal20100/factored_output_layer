import itertools
import unittest

import numpy as np
import scipy

from theano import function
import theano.tensor as T
from theano import config
import theano.tests.unittest_tools as utt
from theano import shared

from op import COST, COST_GRAD, LargeSparseTargets, GpuLargeSparseTargets

import tests.test_cpu_op

class TestLargeSparseTargets(tests.test_cpu_op.TestLargeSparseTargets):
    def setUp(self):
        self.op = GpuLargeSparseTargets

    def test_gpu_opt(self):
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
                     LargeSparseTargets(COST)(V, U, UinvT, Q, H, Y_indexes, 
                                          Y_values, learning_rate))

        assert isinstance(f.maker.fgraph.toposort()[1].op,
                          GpuLargeSparseTargets)

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

        print f.maker.fgraph.toposort()
        assert len(f.maker.fgraph.toposort()) == 3
        assert isinstance(f.maker.fgraph.toposort()[1].op,
                          self.op)
        assert f.maker.fgraph.toposort()[1].op.what_to_output == COST_GRAD
