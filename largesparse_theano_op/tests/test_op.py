from op import LargeSparseTargets

from theano import function
import theano.tensor as T
from theano import config
import theano.tests.unittest_tools as utt
from theano import shared

import numpy as np

#def test_gradient():
#    d = 7
#    D = 30
#    K = 3
#    m = 5
#
#    V_mat = np.asarray(np.random.rand(D, d), dtype=config.floatX)
#    U_mat = np.eye(d, dtype=config.floatX)
#    UinvT_mat = np.eye(d, dtype=config.floatX)
#    Q_mat = np.dot( V_mat.T, V_mat )
#
#    V = shared(V_mat)
#    U = shared(U_mat)
#    UinvT = shared(UinvT_mat)
#    Q = shared(Q_mat)
#
#    H = T.matrix()
#    Y_indexes = T.imatrix()
#    Y_values = T.matrix()
#
#    learning_rate = 0.1
#
#    print V.ndim
#    print V.dtype
#
#    node = LargeSparseTargets(2)(V, U, UinvT, Q, H, Y_indexes, 
#                          Y_values, learning_rate)
#
#    isinstance(T.grad(node, H).owner, LargeSparseTargets)
#
#    
#
#    f = function([H, Y_indexes, Y_values], 
#                 )
#
#    H_mat = np.asarray(np.random.rand(m, d), dtype=config.floatX)
#    Y_indexes_mat = np.asarray(np.random.rand(m, K), dtype=np.int32)
#    Y_values_mat = np.asarray(np.random.rand(m, K), dtype=config.floatX)
 

def test_large_sparse_targets():
    d = 7
    D = 30
    K = 3
    m = 5

    V_mat = np.asarray(np.random.rand(D, d), dtype=config.floatX)
    U_mat = np.eye(d, dtype=config.floatX)
    UinvT_mat = np.eye(d, dtype=config.floatX)
    Q_mat = np.dot( V_mat.T, V_mat )

    V = shared(V_mat)
    U = shared(U_mat)
    UinvT = shared(UinvT_mat)
    Q = shared(Q_mat)

    H = T.matrix()
    Y_indexes = T.imatrix()
    Y_values = T.matrix()

    learning_rate = 0.1

    print V.ndim
    print V.dtype

    f = function([H, Y_indexes, Y_values], 
                 LargeSparseTargets(2)(V, U, UinvT, Q, H, Y_indexes, 
                                      Y_values, learning_rate))

    H_mat = np.asarray(np.random.rand(m, d), dtype=config.floatX)
    Y_indexes_mat = np.asarray(np.random.rand(m, K), dtype=np.int32)
    Y_values_mat = np.asarray(np.random.rand(m, K), dtype=config.floatX)
    
    #inp = np.asarray(np.random.rand(5, 4), dtype=config.floatX)
    #trash = np.zeros((1, 1)).astype(config.floatX)

    #print inp.shape
    #print inp.dtype

    out = f(H_mat, Y_indexes_mat, Y_values_mat)
    print out
    print type(out), type(out[1])

    # utt.assert_allclose(V_mat+1, out[1])

    # Faire planter test pour afficher les printf
    assert(False)


    # V = T.matrix()
    # U = T.matrix()
    # UinvT = T.matrix()
    # Q = T.matrix()


    # H = T.matrix()
    # Y_indexes = T.imatrix()
    # Y_values = T.matrix()

    # learning_rate = 0.1

    # print V.ndim
    # print V.dtype

    # f = function([V, U, UinvT, Q, H, Y_indexes, Y_values], 
    #              LargeSparseTargets()(V, U, UinvT, Q, H, Y_indexes, 
    #                                   Y_values, learning_rate))
    # d = 10
    # D = 30
    # K = 3
    # m = 64

    # V_mat = np.asarray(np.random.rand(D, d), dtype=config.floatX)
    # U_mat = np.eye(d, dtype=config.floatX)
    # UinvT_mat = np.eye(d, dtype=config.floatX)
    # Q_mat = np.dot( V_mat.T, V_mat )

    # H_mat = np.asarray(np.random.rand(m, d), dtype=config.floatX)
    # Y_indexes_mat = np.asarray(np.random.rand(m, K), dtype=np.int32)
    # Y_values_mat = np.asarray(np.random.rand(m, K), dtype=config.floatX)
    
    # #inp = np.asarray(np.random.rand(5, 4), dtype=config.floatX)
    # #trash = np.zeros((1, 1)).astype(config.floatX)

    # #print inp.shape
    # #print inp.dtype

    # out = f(V_mat, U_mat, UinvT_mat, Q_mat, H_mat, Y_indexes_mat, Y_values_mat)
    # print out
    # print type(out), type(out[1])
    # utt.assert_allclose(inp+1, out[1])

    # assert(False)
