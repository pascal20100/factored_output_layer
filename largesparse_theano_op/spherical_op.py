# spherical_op.py
# Copyright (C) 2016 Pascal Vincent and Universite de Montreal
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright
#     notice, this list of conditions and the following disclaimer in the
#     documentation and/or other materials provided with the distribution.
#
#  3. The name of the authors may not be used to endorse or promote
#     products derived from this software without specific prior written
#     permission.
#
# THIS SOFTWARE IS PROVIDED BY THE AUTHORS ``AS IS'' AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN
# NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
# TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


from collections import OrderedDict

import theano
from theano import Apply
from theano.gof import COp, toolbox, TopoOptimizer, Optimizer, local_optimizer
from theano.tensor import as_tensor_variable
import theano.tensor as T

from theano.sandbox.cuda.basic_ops import as_cuda_ndarray_variable, host_from_gpu
from theano.sandbox.cuda.type import CudaNdarrayType

from theano.compile import optdb

from theano.gradient import NullType

from theano.tensor.blas import ldflags

import numpy as np

def numpy_recompute_W(V,UT,omega):
    W = np.dot(V,UT.T) + omega.reshape( (1,omega.shape[0]) )
    return W

def numpy_fprop(W, HT, KindexesT):
    # Computes and returns AT,q,s computed with numpy

    D,d  = W.shape
    m,d = HT.shape
    m,K = KindexesT.shape
    
    oT = np.dot(HT,W.T)
    AT = np.zeros((m,K), dtype=HT.dtype)
    q = np.zeros((m,), dtype=HT.dtype)
    s = np.zeros((m,), dtype=HT.dtype)
    for i in xrange(m):
        o_i = oT[i]
        q[i] = np.dot(o_i, o_i)
        s[i] = np.sum(o_i)
        for k in xrange(K):
            j = KindexesT[i,k]
            AT[i,k] = o_i[j]

    return [AT,q,s]

def numpy_bprop_update(W, HT, KindexesT,
                       grad_AT, grad_q, grad_s,
                       eta
                       ):
    # The call computes and returns grad_HT, new_W
    D,d  = W.shape
    m,d = HT.shape
    m,K = KindexesT.shape

    # recompute oT
    oT = np.dot(HT,W.T)

    # compute grad_oT
    grad_oT = np.zeros((m,D), HT.dtype)
    grad_oT += grad_s.reshape((m,1))
    grad_oT += oT*((grad_q+grad_q).reshape((m,1)))

    for i in xrange(m):
        for k in xrange(K):
            j = KindexesT[i,k]
            grad_oT[i,j] = grad_oT[i,j] + grad_AT[i,k]

    # compute the gradients for backprop
    grad_HT = np.dot(grad_oT,W)
    new_W = W - eta * np.dot(grad_oT.T, HT)

    return grad_HT, new_W


##
def check_tensor_variables_ndim_and_dtype(ndim, dtype, varnames, vardict):
    """Checks whether the variables named in the varnames list,
    and looked for in the vardict dictionary
    are theano tensor variables of the given ndim and dtype
    Otherwise a TypeError is raised.
    """

    for varname in varnames:
        try:
            var = vardict[varname]
            if var.ndim != ndim:
                raise TypeError(varname + " must be a tensor variable of ndim="+str(ndim) )
            if var.dtype != dtype:
                raise TypeError(varname + " is expected to be of dtype " + str(dtype) + ", not "+ str(var.dtype))
        except KeyError:
            raise KeyError(varname + " not in vardict")
        except AttributeError:
            raise TypeError(varname + " must be a tensor variable, but it has no attribute dtype or ndim")
        

class FactoredSphericalOp(COp):

    __props__ = ("eta", 
                 "invup_mode",
                 "stabilize_period",
                 "debug_print")

    func_file = "./factored_spherical_op.cc"
    # func_name = "APPLY_SPECIFIC(factored_spherical_op_fprop)"

    def __init__(self,
                 eta, # the learning rate: can be a tensor variable
                 invup_mode = 4,
                 stabilize_period = 10,    # every how many updates to singular_stabilize
                 debug_print = 0
                 ):

        self.eta = eta
        self.invup_mode = invup_mode
        self.stabilize_period = stabilize_period
        self.debug_print = debug_print

        template_params = "<" + str(invup_mode) \
                          + ", " + str(stabilize_period) \
                          + ", " + str(debug_print) + ">"
        self.func_name = "APPLY_SPECIFIC(factored_spherical_op_fprop)"+template_params

        super(FactoredSphericalOp, self).__init__(self.func_file,
                                                 self.func_name)

    def c_libraries(self):
        return ldflags()

    def c_compile_args(self):
        return ldflags(libs=False, flags=True)

    def c_lib_dirs(self):
        return ldflags(libs=False, libs_dir=True)

    def c_header_dirs(self):
        return ldflags(libs=False, include_dir=True)

    def c_code_cache_version(self):
        return None

    def make_node(self,
                  # model parameters and bookkeeping variables
                  V, UT, Uinv, QT, omega, w_bar,
                  # minibatch value inputs
                  HT, KindexesT
                  ):
        """outputs will be: AT, q, s, work_d, work_m"""

        V = as_tensor_variable(V)
        UT = as_tensor_variable(UT)
        Uinv = as_tensor_variable(Uinv)
        QT = as_tensor_variable(QT)
        omega = as_tensor_variable(omega)
        w_bar = as_tensor_variable(w_bar)
        
        HT = as_tensor_variable(HT)
        KindexesT = as_tensor_variable(KindexesT)

        params = [V, UT, Uinv, QT, omega, w_bar, HT, KindexesT]

        # make sure parameters are either all of dtype float32 or all of dtype float64 (except for Kindexes which are integers)
        elem_type = V.dtype
        if elem_type != "float32" and elem_type != "float64":
            raise TypeError("FactoredSphericalOp parameter V must have dtype of float32 or float64")

        check_tensor_variables_ndim_and_dtype(2, elem_type, ["V", "UT", "Uinv", "QT", "HT"], locals() )
        check_tensor_variables_ndim_and_dtype(1, elem_type, ["omega", "w_bar"], locals() )
        check_tensor_variables_ndim_and_dtype(2, "int32", ["KindexesT"], locals() )

        # Now properly set up outputs to compute: AT, q, s
        outputs = [ T.matrix('AT', elem_type), T.vector('q', elem_type), T.vector('s', elem_type),
                    T.matrix('work_d', elem_type), T.matrix('work_m', elem_type) ]
        
        return Apply(self, params, outputs)

    
 
#    def perform(self, node, inputs, output_storage):

    def grad(self, inputs, output_gradients):
        V, UT, Uinv, QT, omega, w_bar, HT, KindexesT = inputs

        eta = self.eta

        AT, q, s, work_d, work_m = FactoredSphericalOp(
            eta = eta, 
            invup_mode = self.invup_mode,
            stabilize_period = self.stabilize_period,  
            debug_print = self.debug_print
            )(*inputs)
        
        grad_AT, grad_q, grad_s, grad_work_d, grad_work_m  = output_gradients 

        grads = [NullType("No gradient for \'%s\'" % i.name)() for i in inputs]

        # print "AAAAAAAAA  In grad: self.invup_mode=", self.invup_mode

        grads[6] = FactoredSphericalOpBpropUpdate(
            self.invup_mode,
            self.stabilize_period,
            self.debug_print
            )(
                  # model parameters and bookkeeping variables
            V, UT, Uinv, QT, omega, w_bar,
            # minibatch value inputs
            HT, KindexesT,
            # workspace
            work_d, work_m,
            #  minibatch gradient inputs
            grad_AT, grad_q, grad_s,
            # learning rate
            eta)

        return grads


class FactoredSphericalOpBpropUpdate(COp):

    __props__ = ("invup_mode",
                 "stabilize_period",
                 "debug_print")

    func_file = "./factored_spherical_op.cc"
    # func_name = "APPLY_SPECIFIC(factored_spherical_op_fprop)"

    def __init__(self,
                 invup_mode = 4,
                 stabilize_period = 10,    # every how many updates to singular_stabilize
                 debug_print = 0
                 ):
        
        self.invup_mode = invup_mode
        self.stabilize_period = stabilize_period
        self.debug_print = debug_print

        template_params = "<" + str(invup_mode) \
                          + ", " + str(stabilize_period) \
                          + ", " + str(debug_print) + ">"
        self.func_name = "APPLY_SPECIFIC(factored_spherical_op_bprop_update)"+template_params

        super(FactoredSphericalOpBpropUpdate, self).__init__(self.func_file,
                                                 self.func_name)

    def c_libraries(self):
        return ldflags()

    def c_compile_args(self):
        return ldflags(libs=False, flags=True)

    def c_lib_dirs(self):
        return ldflags(libs=False, libs_dir=True)

    def c_header_dirs(self):
        return ldflags(libs=False, include_dir=True)

    def c_code_cache_version(self):
        return None

    def make_node(self,
                  # model parameters and bookkeeping variables
                  V, UT, Uinv, QT, omega, w_bar,
                  # minibatch value inputs
                  HT, KindexesT,
                  # workspace
                  work_d, work_m,
                  #  minibatch gradient inputs
                  grad_AT, grad_q, grad_s,
                  # learning rate
                  eta
                  ):
        """output will be: grad_HT """

        V = as_tensor_variable(V)
        UT = as_tensor_variable(UT)
        Uinv = as_tensor_variable(Uinv)
        QT = as_tensor_variable(QT)
        omega = as_tensor_variable(omega)
        w_bar = as_tensor_variable(w_bar)
        
        HT = as_tensor_variable(HT)
        KindexesT = as_tensor_variable(KindexesT)
        work_d = as_tensor_variable(work_d)
        work_m = as_tensor_variable(work_m)

        grad_AT = as_tensor_variable(grad_AT)
        grad_q = as_tensor_variable(grad_q)
        grad_s = as_tensor_variable(grad_s)
        eta = as_tensor_variable(eta)

        params = [V, UT, Uinv, QT, omega, w_bar,
                  HT, KindexesT, work_d, work_m,
                  grad_AT, grad_q, grad_s, eta]

        # make sure parameters are either all of dtype float32 or all of dtype float64 (except for Kindexes which are integers)
        elem_type = V.dtype
        if elem_type != "float32" and elem_type != "float64":
            raise TypeError("FactoredSphericalOp parameter V must have dtype of float32 or float64")

        check_tensor_variables_ndim_and_dtype(0, elem_type, ["eta"], locals() )
        check_tensor_variables_ndim_and_dtype(2, elem_type, ["V", "UT", "Uinv", "QT", "HT", "grad_AT", "work_d", "work_m"], locals() )
        check_tensor_variables_ndim_and_dtype(1, elem_type, ["omega", "w_bar", "grad_q", "grad_s"], locals() )
        check_tensor_variables_ndim_and_dtype(2, "int32", ["KindexesT"], locals() )

        # Now properly set up outputs to compute: grad_HT
        outputs = [ T.fmatrix('grad_HT') ]
        
        return Apply(self, params, outputs)


 
#    def perform(self, node, inputs, output_storage):

    def grad(self, inputs, output_gradients):
        # inputs = V, UT, Uinv, QT, omega, w_bar, HT, KindexesT, grad_AT, grad_q, grad_s, eta
        grads = [NullType("No gradient for \'%s\'" % i.name)() for i in inputs]
        
        return grads




#######################################################
#################     GPU Versions   ##################

class GpuFactoredSphericalOp(COp):

    __props__ = ("eta", 
                 "invup_mode",
                 "stabilize_period",
                 "debug_print")

    func_file = "./factored_spherical_op_gpu.cc"
    # func_name = "APPLY_SPECIFIC(factored_spherical_op_fprop)"

    def __init__(self,
                 eta, # the learning rate: can be a tensor variable
                 invup_mode = 5,
                 stabilize_period = 10,    # every how many updates to singular_stabilize
                 debug_print = 0
                 ):

        self.eta = eta
        self.invup_mode = invup_mode
        self.stabilize_period = stabilize_period
        self.debug_print = debug_print

        template_params = "<" + str(invup_mode) \
                          + ", " + str(stabilize_period) \
                          + ", " + str(debug_print) + ">"
        self.func_name = "APPLY_SPECIFIC(factored_spherical_op_fprop)"+template_params

        super(GpuFactoredSphericalOp, self).__init__(self.func_file,
                                                 self.func_name)

    def c_libraries(self):
        return ldflags()

    def c_compile_args(self):
        return ldflags(libs=False, flags=True)

    def c_lib_dirs(self):
        return ldflags(libs=False, libs_dir=True)

    def c_header_dirs(self):
        return ldflags(libs=False, include_dir=True)

    def c_code_cache_version(self):
        return None

    def make_node(self,
                  # model parameters and bookkeeping variables
                  V, UT, Uinv, QT, omega, w_bar,
                  # minibatch value inputs
                  HT, KindexesT
                  ):
        """outputs will be: AT, q, s, work_d, work_m"""

        # The following are supposed to reside on the GPU
        V = as_cuda_ndarray_variable(V)
        UT = as_cuda_ndarray_variable(UT)
        Uinv = as_cuda_ndarray_variable(Uinv)
        QT = as_cuda_ndarray_variable(QT)
        omega = as_cuda_ndarray_variable(omega)
        w_bar = as_cuda_ndarray_variable(w_bar)
        
        HT = as_cuda_ndarray_variable(HT)

        # This is on GPU
        KindexesT = as_tensor_variable(KindexesT)

        # List of op parameters
        params = [V, UT, Uinv, QT, omega, w_bar, HT, KindexesT]

        # make sure parameters are either all of dtype float32 or all of dtype float64 (except for Kindexes which are integers)
        elem_type = V.dtype
        if elem_type != "float32" and elem_type != "float64":
            raise TypeError("GpuFactoredSphericalOp parameter V must have dtype of float32 or float64")

        check_tensor_variables_ndim_and_dtype(2, elem_type, ["V", "UT", "Uinv", "QT", "HT"], locals() )
        check_tensor_variables_ndim_and_dtype(1, elem_type, ["omega", "w_bar"], locals() )
        check_tensor_variables_ndim_and_dtype(2, "int32", ["KindexesT"], locals() )

        # Now properly set up outputs to compute: AT, q, s
        outputs = [ CudaNdarrayType(broadcastable=(False,False))(), # AT
                    CudaNdarrayType(broadcastable=(False,))(), # q
                    CudaNdarrayType(broadcastable=(False,))(), # s
                    CudaNdarrayType(broadcastable=(False,False))(), # work_d
                    CudaNdarrayType(broadcastable=(False,False))() # work_m
                    ]
        
        return Apply(self, params, outputs)

    
 
#    def perform(self, node, inputs, output_storage):

    def grad(self, inputs, output_gradients):
        V, UT, Uinv, QT, omega, w_bar, HT, KindexesT = inputs

        eta = self.eta

        AT, q, s, work_d, work_m = GpuFactoredSphericalOp(
            eta = eta, 
            invup_mode = self.invup_mode,
            stabilize_period = self.stabilize_period,  
            debug_print = self.debug_print
            )(*inputs)
        
        grad_AT, grad_q, grad_s, grad_work_d, grad_work_m  = output_gradients 

        grads = [NullType("No gradient for \'%s\'" % i.name)() for i in inputs]

        # print "AAAAAAAAA  In grad: self.invup_mode=", self.invup_mode

        grads[6] = GpuFactoredSphericalOpBpropUpdate(
            self.invup_mode,
            self.stabilize_period,
            self.debug_print
            )(
                  # model parameters and bookkeeping variables
            V, UT, Uinv, QT, omega, w_bar,
            # minibatch value inputs
            HT, KindexesT,
            # workspace
            work_d, work_m,
            #  minibatch gradient inputs
            grad_AT, grad_q, grad_s,
            # learning rate
            eta)

        return grads


class GpuFactoredSphericalOpBpropUpdate(COp):

    __props__ = ("invup_mode",
                 "stabilize_period",
                 "debug_print")

    func_file = "./factored_spherical_op_gpu.cc"
    # func_name = "APPLY_SPECIFIC(factored_spherical_op_fprop)"

    def __init__(self,
                 invup_mode = 5,
                 stabilize_period = 10,    # every how many updates to singular_stabilize
                 debug_print = 0
                 ):
        
        self.invup_mode = invup_mode
        self.stabilize_period = stabilize_period
        self.debug_print = debug_print

        template_params = "<" + str(invup_mode) \
                          + ", " + str(stabilize_period) \
                          + ", " + str(debug_print) + ">"
        self.func_name = "APPLY_SPECIFIC(factored_spherical_op_bprop_update)"+template_params

        super(GpuFactoredSphericalOpBpropUpdate, self).__init__(self.func_file,
                                                 self.func_name)

    def c_libraries(self):
        return ldflags()

    def c_compile_args(self):
        return ldflags(libs=False, flags=True)

    def c_lib_dirs(self):
        return ldflags(libs=False, libs_dir=True)

    def c_header_dirs(self):
        return ldflags(libs=False, include_dir=True)

    def c_code_cache_version(self):
        return None

    def make_node(self,
                  # model parameters and bookkeeping variables
                  V, UT, Uinv, QT, omega, w_bar,
                  # minibatch value inputs
                  HT, KindexesT,
                  # workspace
                  work_d, work_m,
                  #  minibatch gradient inputs
                  grad_AT, grad_q, grad_s,
                  # learning rate
                  eta
                  ):
        """output will be: grad_HT """

        # The following are supposed to reside on the GPU
        V = as_cuda_ndarray_variable(V)
        UT = as_cuda_ndarray_variable(UT)
        Uinv = as_cuda_ndarray_variable(Uinv)
        QT = as_cuda_ndarray_variable(QT)
        omega = as_cuda_ndarray_variable(omega)
        w_bar = as_cuda_ndarray_variable(w_bar)
        
        HT = as_cuda_ndarray_variable(HT)

        # This is on CPU
        KindexesT = as_tensor_variable(KindexesT)

        # The following are supposed to reside on the GPU
        work_d = as_cuda_ndarray_variable(work_d)
        work_m = as_cuda_ndarray_variable(work_m)

        grad_AT = as_cuda_ndarray_variable(grad_AT)
        grad_q = as_cuda_ndarray_variable(grad_q)
        grad_s = as_cuda_ndarray_variable(grad_s)

        # This is on CPU
        eta = as_tensor_variable(eta)

        # parametr list
        params = [V, UT, Uinv, QT, omega, w_bar,
                  HT, KindexesT, work_d, work_m,
                  grad_AT, grad_q, grad_s, eta]

        # make sure parameters are either all of dtype float32 or all of dtype float64 (except for Kindexes which are integers)
        elem_type = V.dtype
        if elem_type != "float32" and elem_type != "float64":
            raise TypeError("GpuFactoredSphericalOp parameter V must have dtype of float32 or float64")

        check_tensor_variables_ndim_and_dtype(0, elem_type, ["eta"], locals() )
        check_tensor_variables_ndim_and_dtype(2, elem_type, ["V", "UT", "Uinv", "QT", "HT", "grad_AT", "work_d", "work_m"], locals() )
        check_tensor_variables_ndim_and_dtype(1, elem_type, ["omega", "w_bar", "grad_q", "grad_s"], locals() )
        check_tensor_variables_ndim_and_dtype(2, "int32", ["KindexesT"], locals() )

        # Now properly set up outputs to compute: grad_HT
        outputs = [ CudaNdarrayType(broadcastable=(False,False))() ]
        
        return Apply(self, params, outputs)


 
#    def perform(self, node, inputs, output_storage):

    def grad(self, inputs, output_gradients):
        # inputs = V, UT, Uinv, QT, omega, w_bar, HT, KindexesT, grad_AT, grad_q, grad_s, eta
        grads = [NullType("No gradient for \'%s\'" % i.name)() for i in inputs]
        
        return grads


