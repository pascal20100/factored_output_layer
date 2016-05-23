import itertools
import unittest

import numpy as np
import scipy

from theano import function, function_dump
import theano.tensor as T
from theano import config
import theano.tests.unittest_tools as utt
from theano import shared

import sys

import pdb

# from op import COST, COST_GRAD, LargeSparseTargets
import spherical_op

class TestSphericalOp(unittest.TestCase):
    def setUp(self):
        self.op = spherical_op.GpuFactoredSphericalOp
        self.d = 3
        self.D = 7
        self.K = 2
        self.m = 5
        self.eta = np.float32(0.1)
        # self.invup_mode = 4
        self.invup_mode = 4

    def generate_model_params(self, seed=37):
        d = self.d
        D = self.D
        K = self.K
        m = self.m

        np.random.seed(seed)

        V_mat = np.asarray(np.random.rand(D, d), dtype=config.floatX)
        UT_mat = np.eye(d, dtype=config.floatX)
        Uinv_mat = np.eye(d, dtype=config.floatX)
        omega_vec = np.asarray(np.random.rand(d), dtype=config.floatX)

        W = spherical_op.numpy_recompute_W(V_mat, UT_mat, omega_vec)
        w_bar_vec = W.sum(axis=0)
        QT_mat = np.dot( W.T, W )

        return V_mat, UT_mat, Uinv_mat, QT_mat, omega_vec, w_bar_vec

    def generate_data(self, seed=42):
        d = self.d
        D = self.D
        K = self.K
        m = self.m

        np.random.seed(seed)

        HT_mat = np.asarray(np.random.rand(m, d), dtype=config.floatX)

        KindexesT_mat = np.asarray([ np.random.permutation(D)[0:K] for i in xrange(m) ], dtype=np.int32)
        
        # KindexesT_mat = np.zeros((m,K),dtype=np.int32)
        # for i in xrange(m):
        #     for k in xrange(K):
        #         j = np.random.randint(0,D)
        #         while j in KindexesT_mat[i,0:k]:
        #             j = np.random.randint(0,D)
        #         KindexesT_mat[i,k] = j
        
        grad_AT_mat = np.asarray(np.random.rand(m, K), dtype=config.floatX)
        grad_q_vec = np.asarray(np.random.rand(m), dtype=config.floatX)
        grad_s_vec = np.asarray(np.random.rand(m), dtype=config.floatX)
        
        # return HT_mat, KindexesT_mat, Y_values_mat, Y
        return HT_mat, KindexesT_mat, grad_AT_mat, grad_q_vec, grad_s_vec


    def test_spherical_op(self):
        print >> sys.stderr
        print >> sys.stderr, "================================================="
        print >> sys.stderr, "============  TEST spherical_op  ==========="
        print >> sys.stderr, "================================================="
        
        V_mat, UT_mat, Uinv_mat, QT_mat, omega_vec, w_bar_vec = self.generate_model_params()
        HT_mat, KindexesT_mat, grad_AT_mat, grad_q_vec, grad_s_vec = self.generate_data()            

        V = shared(V_mat)
        UT = shared(UT_mat)
        Uinv = shared(Uinv_mat)
        QT = shared(QT_mat)
        omega = shared(omega_vec)
        w_bar = shared(w_bar_vec)

        HT = T.matrix()
        KindexesT = T.imatrix()
        Y_values = T.matrix()

        eta_var = shared(self.eta)

        # compute AT,q,s with theano (factorized algo)
        f = function([HT, KindexesT], 
                     spherical_op.GpuFactoredSphericalOp(eta_var, invup_mode=self.invup_mode)(V, UT, Uinv, QT, omega, w_bar,
                                                           HT, KindexesT))
        th_AT, th_q, th_s, work_d, work_m  = f(HT_mat, KindexesT_mat)

        # compute AT,q,s with numpy (unfactorized algo)
        W_mat = spherical_op.numpy_recompute_W(V_mat, UT_mat, omega_vec)
        np_AT, np_q, np_s = spherical_op.numpy_fprop(W_mat, HT_mat, KindexesT_mat)

        print >> sys.stderr, "******** theano and numpy AT ********"
        print >> sys.stderr, th_AT
        print >> sys.stderr, "---"
        print >> sys.stderr, np_AT
        print >> sys.stderr, "******** theano and numpy q ********"
        print >> sys.stderr, th_q
        print >> sys.stderr, "---"
        print >> sys.stderr, np_q
        print >> sys.stderr, "******** theano and numpy s ********"
        print >> sys.stderr, th_s
        print >> sys.stderr, "---"
        print >> sys.stderr, np_s

        utt.assert_allclose(th_AT, np_AT)
        utt.assert_allclose(th_q, np_q)
        utt.assert_allclose(th_s, np_s)

    def test_spherical_op_grad(self):
        print >> sys.stderr
        print >> sys.stderr, "================================================="
        print >> sys.stderr, "============  TEST spherical_op_grad  ==========="
        print >> sys.stderr, "================================================="


        V_mat, UT_mat, Uinv_mat, QT_mat, omega_vec, w_bar_vec = self.generate_model_params()
        HT_mat, KindexesT_mat, grad_AT_mat, grad_q_vec, grad_s_vec = self.generate_data()            

        V = shared(V_mat)
        UT = shared(UT_mat)
        Uinv = shared(Uinv_mat)
        QT = shared(QT_mat)
        omega = shared(omega_vec)
        w_bar = shared(w_bar_vec)

        # Fiddling to debug:
        # grad_AT_mat.fill(0.)
        # grad_q_vec.fill(0.)
        # grad_s_vec.fill(0.)
        # grad_AT_mat[0,0] = 10000.

        grad_AT = shared(grad_AT_mat)
        grad_q = shared(grad_q_vec)
        grad_s = shared(grad_s_vec)

        HT = T.matrix()
        KindexesT = T.imatrix()
        eta_var = T.scalar()

        # compute AT,q,s with numpy (unfactorized algo)
        W_mat = spherical_op.numpy_recompute_W(V_mat, UT_mat, omega_vec)
        # numpy_out = spherical_op.numpy_fprop(W_mat, HT_mat, KindexesT_mat)
        numpy_grad_HT, numpy_new_W = spherical_op.numpy_bprop_update(W_mat, HT_mat, KindexesT_mat,
                                                                     grad_AT_mat, grad_q_vec, grad_s_vec,
                                                                     self.eta)

        # compute grad_HT with theano (factorized algo)
        AT, q, s, work_d, work_m = spherical_op.GpuFactoredSphericalOp(eta_var, invup_mode=self.invup_mode)(V, UT, Uinv, QT, omega, w_bar,
                                                                         HT, KindexesT)

        linked_grad_s = grad_s + 1e-9*s
        grad_HT = spherical_op.GpuFactoredSphericalOpBpropUpdate(invup_mode=self.invup_mode)(
            V, UT, Uinv, QT, omega, w_bar,
            HT, KindexesT,
            work_d, work_m,
            grad_AT, grad_q, linked_grad_s,
            eta_var )

        # pdb.set_trace()
        g = function([HT, KindexesT, eta_var],[grad_HT])
        # g = function_dump("my_theano_function.dump",[HT, KindexesT, eta],[grad_HT])
        
        theano_grad_HT, = g(HT_mat, KindexesT_mat, self.eta)

        
        # # compute AT,q,s with theano (factorized algo)
        # f = function([HT, KindexesT], 
        #              spherical_op.GpuFactoredSphericalOp(eta)(V, UT, Uinv, QT, omega, w_bar,
        #                                                    HT, KindexesT))
        # theano_AT, theano_q, theano_s = f(HT_mat, KindexesT_mat)

        # g = function([HT, KindexesT, grad_AT, grad_q, grad_s, eta ], 
        #              spherical_op.GpuFactoredSphericalOpBpropUpdate()(V, UT, Uinv, QT, omega, w_bar,
        #                                                            HT, KindexesT,
        #                                                            grad_AT, grad_q, grad_s,
        #                                                            eta ))
        # theano_grad_HT = g(HT_mat, KindexesT_mat, grad_AT_mat, grad_q_vec, grad_s_vec, self.eta)


        print >> sys.stderr, "******** KindexesT ********"
        print >> sys.stderr, KindexesT_mat

        print >> sys.stderr, "******** numpy grad_HT ********"
        print >> sys.stderr, numpy_grad_HT
        print >> sys.stderr, "******** theano grad_HT ********"
        print >> sys.stderr, np.asarray(theano_grad_HT)

        
        print >> sys.stderr, "******** old W ********"
        print >> sys.stderr, W_mat
        print >> sys.stderr, "******** numpy new W ********"
        print >> sys.stderr, numpy_new_W
        print >> sys.stderr, "******** theano new W ********"
        new_W_mat = spherical_op.numpy_recompute_W(V.get_value(), UT.get_value(), omega.get_value())
        print >> sys.stderr, new_W_mat
        print >> sys.stderr, "******** consistency of op's new w_bar and W ********"
        print >> sys.stderr, "w_bar before update: ", w_bar_vec
        print >> sys.stderr, "sum of old W: ", W_mat.sum(axis=0)        
        print >> sys.stderr, "w_bar after update:", w_bar.get_value()
        print >> sys.stderr, "sum of numpy new W:", new_W_mat.sum(axis=0)        
        print >> sys.stderr, "******** consistency of op's new QT and W ********"
        print >> sys.stderr, "QT before update:"
        print >> sys.stderr, QT_mat
        print >> sys.stderr, "QT after update: (consider only upper triangular part)"
        print >> sys.stderr, QT.get_value()
        print >> sys.stderr, "numpy new W.T W:"
        print >> sys.stderr, np.dot(new_W_mat.T, new_W_mat)


        #utt.assert_allclose(theano_out[0], numpy_out[0])
        #utt.assert_allclose(theano_out[1], numpy_out[1])
        #utt.assert_allclose(theano_out[2], numpy_out[2])


    def test_spherical_op_take_gradient(self):
        print >> sys.stderr        
        print >> sys.stderr, "================================================="
        print >> sys.stderr, "====== TEST spherical_op_take_gradient ==========="
        print >> sys.stderr, "================================================="

        V_mat, UT_mat, Uinv_mat, QT_mat, omega_vec, w_bar_vec = self.generate_model_params()
        HT_mat, KindexesT_mat, grad_AT_mat, grad_q_vec, grad_s_vec = self.generate_data()            

        V = shared(V_mat)
        UT = shared(UT_mat)
        Uinv = shared(Uinv_mat)
        QT = shared(QT_mat)
        omega = shared(omega_vec)
        w_bar = shared(w_bar_vec)

        # Fiddling to debug:
        # grad_AT_mat.fill(0.)
        # grad_q_vec.fill(0.)
        # grad_s_vec.fill(0.)
        # grad_AT_mat[0,0] = 10000.

        grad_AT = shared(grad_AT_mat)
        grad_q = shared(grad_q_vec)
        grad_s = shared(grad_s_vec)

        HT = T.matrix()
        KindexesT = T.imatrix()
        eta_var = T.scalar()
        # eta = shared(self.eta)


        # compute AT,q,s with numpy (unfactorized algo)
        W_mat = spherical_op.numpy_recompute_W(V_mat, UT_mat, omega_vec)
        # numpy_out = spherical_op.numpy_fprop(W_mat, HT_mat, KindexesT_mat)
        numpy_grad_HT, numpy_new_W = spherical_op.numpy_bprop_update(W_mat, HT_mat, KindexesT_mat,
                                                                     grad_AT_mat, grad_q_vec, grad_s_vec,
                                                                     self.eta)

        # compute grad_HT with theano (factorized algo)
        AT, q, s, work_d, work_m = spherical_op.GpuFactoredSphericalOp(eta_var, invup_mode=self.invup_mode)(V, UT, Uinv, QT, omega, w_bar,
                                                                         HT, KindexesT)

        def z_loss(AT, q, s, eps=1e-12):
            D = self.D
            mu = s / D
            sigma = T.sqrt((q / D) - mu**2)
            c = T.nnet.softplus((mu - AT[:, 0]) / (sigma + eps))
            return c.sum()

        def simple_loss(AT, q, s, eps=1e-5):
            D = self.D
            mu = s / D
            sigma2 = (q / D) - mu**2 + eps
            c = sigma2 + 0*AT[:,0]
            return c.sum()

        def simplest_loss(AT, q, s, eps=1e-5):
            L = -AT[:,0]+0.7*s+0.3*q
            return L.sum()


        L = z_loss(AT,q,s)
        L = simplest_loss(AT,q,s)

        # grad_HT = theano.grad(L, [sub] + params)
        grad_HT = T.grad(L, wrt=HT)

        # linked_grad_s = grad_s + 1e-9*s
        # grad_HT = spherical_op.GpuFactoredSphericalOpBpropUpdate(invup_mode=self.invup_mode)(
        #     V, UT, Uinv, QT, omega, w_bar,
        #     HT, KindexesT,
        #     work_d, work_m,
        #     grad_AT, grad_q, linked_grad_s,
        #     eta )

        # pdb.set_trace()
        g = function([HT, KindexesT, eta_var],[AT,q,s,L,grad_HT])
        # g = function_dump("my_theano_function.dump",[HT, KindexesT, eta_var],[grad_HT])
        
        theano_AT, theano_q, theano_s, theano_L, theano_grad_HT = g(HT_mat, KindexesT_mat, self.eta)

        
        print >> sys.stderr, "******** KindexesT ********"
        print >> sys.stderr, KindexesT_mat

        print >> sys.stderr, "******** theano AT ********"
        # print >> sys.stderr, dir(theano_AT)
        print >> sys.stderr, np.asarray(theano_AT)
        print >> sys.stderr, "******** theano q  ********"
        print >> sys.stderr, np.asarray(theano_q)
        print >> sys.stderr, "******** theano s  ********"
        print >> sys.stderr, np.asarray(theano_s)
        print >> sys.stderr, "******** theano L  ********"
        print >> sys.stderr, theano_L

        #print >> sys.stderr, "******** numpy grad_HT ********"
        #print >> sys.stderr, numpy_grad_HT
        print >> sys.stderr, "******** theano grad_HT ********"
        print >> sys.stderr, theano_grad_HT

        
        print >> sys.stderr, "******** old W ********"
        print >> sys.stderr, W_mat
        #print >> sys.stderr, "******** numpy new W ********"
        #print >> sys.stderr, numpy_new_W
        print >> sys.stderr, "******** theano new W ********"
        new_W_mat = spherical_op.numpy_recompute_W(V.get_value(), UT.get_value(), omega.get_value())
        print >> sys.stderr, new_W_mat
        print >> sys.stderr, "******** consistency of op's new w_bar and W ********"
        print >> sys.stderr, "w_bar before update: ", w_bar_vec
        print >> sys.stderr, "sum of old W: ", W_mat.sum(axis=0)        
        print >> sys.stderr, "w_bar after update:", w_bar.get_value()
        print >> sys.stderr, "sum of numpy new W:", new_W_mat.sum(axis=0)        
        print >> sys.stderr, "******** consistency of op's new QT and W ********"
        print >> sys.stderr, "QT before update:"
        print >> sys.stderr, QT_mat
        print >> sys.stderr, "QT after update: (consider only upper triangular part)"
        print >> sys.stderr, QT.get_value()
        print >> sys.stderr, "numpy new W.T W:"
        print >> sys.stderr, np.dot(new_W_mat.T, new_W_mat)


        #utt.assert_allclose(theano_out[0], numpy_out[0])
        #utt.assert_allclose(theano_out[1], numpy_out[1])
        #utt.assert_allclose(theano_out[2], numpy_out[2])



    def runTest(self):
        self.setUp()
        # self.test_spherical_op()
        self.test_spherical_op_grad()
        self.test_spherical_op_take_gradient()

if __name__ == '__main__':
    optest = TestSphericalOp()
    optest.runTest()
