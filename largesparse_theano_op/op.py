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
        

COST = 0
GRAD = 1
COST_GRAD = 2


class LargeSparseTargets(COp):

    __props__ = ("what_to_output", )

    func_file = "./largesparsetargets.cc"
    # func_name = "APPLY_SPECIFIC(largesparsetargets)"

    def __init__(self, what_to_output=0):
        """what_to_output specifies what outputs the op will compute and return:
        0: means compute and output the loss only
        1: means compute and output the grad_H only
        2: means compute and output both the loss and grad_H
        """

        self.what_to_output = what_to_output
      
        if what_to_output==COST:
            self.func_name = "APPLY_SPECIFIC(largesparsetargets_0)"
        elif what_to_output==GRAD:
            self.func_name = "APPLY_SPECIFIC(largesparsetargets_1)"
        elif what_to_output==COST_GRAD:
            self.func_name = "APPLY_SPECIFIC(largesparsetargets)"
        else:
            raise ValueError("Invalid value for what_to_output: must be 0,1, or 2")
        
        super(LargeSparseTargets, self).__init__(self.func_file,
                                                 self.func_name)

    def c_code_cache_version(self):
        return None

    def make_node(self, V, U, UinvT, Q, H, Y_indexes, Y_values, learning_rate, 
                  use_qtilde=0, use_lower=1, invup_mode=1,
                  stabilize_period=10, unfactorize_period=100,debug_print=0):

        print "invup_mode=",invup_mode

        V = as_tensor_variable(V)
        U = as_tensor_variable(U)
        UinvT = as_tensor_variable(UinvT)
        Q = as_tensor_variable(Q)
        H = as_tensor_variable(H)
        Y_indexes = as_tensor_variable(Y_indexes)
        Y_values = as_tensor_variable(Y_values)
        learning_rate = as_tensor_variable(learning_rate)
        use_qtilde = as_tensor_variable(use_qtilde)
        use_lower = as_tensor_variable(use_lower)
        invup_mode = as_tensor_variable(invup_mode)
        stabilize_period = as_tensor_variable(stabilize_period)
        unfactorize_period = as_tensor_variable(unfactorize_period)
        debug_print = as_tensor_variable(debug_print)

        params = [V, U, UinvT, Q, H, Y_indexes, Y_values, learning_rate,
                  use_qtilde, use_lower, invup_mode, stabilize_period,
                  unfactorize_period, debug_print]

        # make sure parameters are either all of dtype float32 or all of dtype float64 (except for Y_indexes which are integers)
        elem_type = V.dtype
        if elem_type != "float32" and elem_type != "float64":
            raise TypeError("LargeSparseTargets parameter V must have dtype of float32 or float64")

        check_tensor_variables_ndim_and_dtype(0, elem_type, ["learning_rate"], locals() )
        check_tensor_variables_ndim_and_dtype(2, elem_type, ["V", "U", "UinvT", "Q", "H", "Y_values"], locals() )
        check_tensor_variables_ndim_and_dtype(2, "int32", ["Y_indexes"], locals() )

        # Now properly set up outputs to compute
        if self.what_to_output==0: # output scalar cost
            outputs = [ T.scalar(elem_type) ]
        elif self.what_to_output==1: # output grad_H
            outputs = [ T.matrix(elem_type) ]
        elif self.what_to_output==2: # output cost and grad_H
            outputs = [ T.scalar(elem_type), T.matrix(elem_type) ]
        else:
            raise ValueError("Invalid value for what_to_output: must be 0,1, or 2")
        
        return Apply(self, params, outputs)


        # return Apply(self, params, [T.scalar(V.dtype), T.matrix(V.dtype)])

#    def perform(self, node, inputs, output_storage):
#        V = inputs[0]
#        U = inputs[1]
#        UinvT = inputs[2]
#        Q = inputs[3]
#        H = inputs[4]
#        Y_indexes = inputs[5]
#        Y_values = inputs[6]
#        learning_rate = inputs[7]
#
#        L = output_storage[0]
#        grad = output_storage[1]
#
#        h_hat = T.dot(Q, H)
#        y_hat = T.dot(U.transpose(),T.dot(V.transpose(),y))
#
#        z_hat = h_hat - y_hat
#
#        L[0] = T.dot(h.transpose(),h_hat) - 2*T.dot(h.transpose(),y_hat) \
#               + T.dot(y.transpose(), y)
#
#        grad[0] = 2*z_hat
#
#        denom = 1. - 2.*learning_rate*T.sum(T.square(h))
#
#        U = U - 2*learning_rate*T.dot(T.dot(U,h),h.transpose())
#
#        UinvT = UinvT + 2*learning_rate/denom * \
#                        T.dot(T.dot(UinvT,h), h.transpose())
#
#        V = V + 2*learning_rate*T.dot(y, T.dot(UinvT, h).transpose())
#
#        # are values really updated?
#        inputs[0] = V
#        inputs[1] = U
#        inputs[2] = UinvT

    def grad(self, inputs, output_gradients):
        # inputs = [V, U, UinvT, Q, H, Y_indexes, Y_values, learning_rate]

        grads = [NullType("No gradient for \'%s\'" % i.name)() for i in inputs]
        grads[4] = LargeSparseTargets(what_to_output=1)(*inputs)

        return grads

            
class LargeSparseTargetsMaintainance(COp):

    # maintainance_op codes
    SANITY_CHECK_DIMENSIONS = 0
    PRINT_SCALES = 1
    CONSISTENCY_CHECK = 2
    RENORMALIZE_VT = 3
    SINGULAR_STABILIZE = 4
    UNFACTORIZE = 5

    __props__ = ()

    func_file = "./largesparsetargets.cc"
    func_name = "APPLY_SPECIFIC(largesparsetargets_maintainance)"

    def make_node(self, maintainance_op, V, U, UinvT, Q):
        maintainance_op = as_tensor_variable(maintainance_op)        
        V = as_tensor_variable(V)
        U = as_tensor_variable(U)
        UinvT = as_tensor_variable(UinvT)
        Q = as_tensor_variable(Q)

        params = [maintainance_op, V, U, UinvT, Q]

        # make sure parameters are either all of dtype float32 or all of dtype float64 (except for Y_indexes which are integers)
        elem_type = V.dtype
        if elem_type != "float32" and elem_type != "float64":
            raise TypeError("Parameter V must have dtype of float32 or float64")

        check_tensor_variables_ndim_and_dtype(0, "int32", ["maintainance_op"], locals() )
        check_tensor_variables_ndim_and_dtype(2, elem_type, ["V", "U", "UinvT", "Q"], locals() )

        # Now properly set up outputs to compute
        outputs = []        
        return Apply(self, params, outputs)



class GpuLargeSparseTargets(COp):
    __props__ = ("what_to_output", )

    func_file = "./largesparsetargets_gpu.cc"
    # func_name = "APPLY_SPECIFIC(largesparsetargets)"

    def __init__(self, what_to_output=0):
        """what_to_output specifies what outputs the op will compute and return:
        0: means compute and output the loss only
        1: means compute and output the grad_H only
        2: means compute and output both the loss and grad_H
        """

        self.what_to_output = what_to_output

        if what_to_output==0:
            self.func_name = "APPLY_SPECIFIC(largesparsetargets_gpu_0)"
        elif what_to_output==1:
            self.func_name = "APPLY_SPECIFIC(largesparsetargets_gpu_1)"
        elif what_to_output==2:
            self.func_name = "APPLY_SPECIFIC(largesparsetargets_gpu)"
        else:
            raise ValueError("Invalid value for what_to_output: must be 0,1, or 2")
        
        super(GpuLargeSparseTargets, self).__init__(self.func_file,
                                                    self.func_name)

    def c_header_dirs(self):
        return ["/home/bouthilx/projects/pvprojects/2014/LargeSparseTargets/"]

    def c_code_cache_version(self):
        return None

    def make_node(self, V, U, UinvT, Q, H, Y_indexes, Y_values, learning_rate,
                  use_qtilde=0, use_lower=1, invup_mode=1,
                  stabilize_period=10, unfactorize_period=100,debug_print=0):

        # The following are supposed to reside on the GPU
        V = as_cuda_ndarray_variable(V)
        U = as_cuda_ndarray_variable(U)
        UinvT = as_cuda_ndarray_variable(UinvT)
        Q = as_cuda_ndarray_variable(Q)
        H = as_cuda_ndarray_variable(H)

        # The following are on the CPU
        Y_indexes = as_tensor_variable(Y_indexes)
        Y_values = as_tensor_variable(Y_values)
        learning_rate = as_tensor_variable(learning_rate)
        use_qtilde = as_tensor_variable(use_qtilde)
        use_lower = as_tensor_variable(use_lower)
        invup_mode = as_tensor_variable(invup_mode)
        stabilize_period = as_tensor_variable(stabilize_period)
        unfactorize_period = as_tensor_variable(unfactorize_period)
        debug_print = as_tensor_variable(debug_print)

        # print "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
        # for k,v in locals().items():
        #     print k,':',type(v)
        # print "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"

        params = [V, U, UinvT, Q, H, Y_indexes, Y_values, learning_rate,
                  use_qtilde, use_lower, invup_mode, stabilize_period,
                  unfactorize_period, debug_print]

        # make sure parameters are either all of dtype float32 or all of dtype float64 (except for Y_indexes which are integers)
        elem_type = V.dtype
        if elem_type != "float32" and elem_type != "float64":
            raise TypeError("LargeSparseTargets parameter V must have dtype of float32 or float64")

        check_tensor_variables_ndim_and_dtype(0, elem_type, ["learning_rate"], locals() )
        check_tensor_variables_ndim_and_dtype(2, elem_type, ["V", "U", "UinvT", "Q", "H", "Y_values"], locals() )
        check_tensor_variables_ndim_and_dtype(2, "int32", ["Y_indexes"], locals() )

        # T.matrix(elem_type)
            
        # Now properly set up outputs to compute
        if self.what_to_output==0: # output scalar cost
            outputs = [ T.scalar(elem_type) ]
        elif self.what_to_output==1: # output grad_H
            outputs = [ CudaNdarrayType(broadcastable=(False,False))() ]
        elif self.what_to_output==2: # output cost and grad_H
            outputs = [ T.scalar(elem_type), CudaNdarrayType(broadcastable=(False,False))() ]
        else:
            raise ValueError("Invalid value for what_to_output: must be 0,1, or 2")
        
        return Apply(self, params, outputs)


        # return Apply(self, params, [T.scalar(V.dtype), T.matrix(V.dtype)])

    def grad(self, inputs, output_gradients):
        # inputs = [V, U, UinvT, Q, H, Y_indexes, Y_values, learning_rate]

        grads = [NullType("No gradient for \'%s\'" % i.name)() for i in inputs]
        grads[4] = LargeSparseTargets(what_to_output=1)(*inputs)

        # TODO: QUESTION should the above be GPULargeSparseTargets instead???????

        return grads




class MergeLargeSparseTargetOps(Optimizer):
    """
        TODO: WRITE
    """

    def apply(self, fgraph):

        def is_grad_of_op(n1, n2):
            if n1.op.what_to_output != GRAD:
                return False
            if n2.op.what_to_output != COST:
                return False

            if not all(n1i == n2i for n1i, n2i in zip(n1.inputs, n2.inputs)):
                return False

            return True

        fnodes = []
        gnodes = []
        for node in fgraph.toposort():
            if isinstance(node.op, LargeSparseTargets):
                if node.op.what_to_output == COST:
                    fnodes.append((node, []))
                elif node.op.what_to_output == GRAD:
                    gnodes.append(node)

        for gnode in gnodes:
            for fnode, fgnodes in fnodes:
                if is_grad_of_op(gnode, fnode):
                    fgnodes.append(gnode)

        for fnode, gnodes in fnodes:
            if len(gnodes) == 0:
                continue

            a = LargeSparseTargets(what_to_output=2).make_node(*fnode.inputs)
            f, g = a.outputs

            z = fnode.outputs[0]
            fgraph.replace_validate(z, f, "replace by a cost+grad op")

            for gnode in gnodes:
                z = gnode.outputs[0]
                fgraph.replace_validate(z, g, "replace by a cost+grad op")


mergelst = MergeLargeSparseTargetOps()
#optdb['specialize'].register('merge_large_sparse_target_ops', mergelst, 'fast_run')

optdb.register("global_large_sparse_targets_merge", mergelst, 48.5, "fast_run")

# add CPU TO GPU merge
#@register_specialize
#@local_optimizer([LargeSparseTargets])
def local_large_sparse_targets_gpu(node):
    if not isinstance(node.op, LargeSparseTargets) or theano.config.device == "cpu":
        return False

    if node.op.what_to_output == 0:
        return [GpuLargeSparseTargets(node.op.what_to_output)(*node.inputs)]
    elif node.op.what_to_output == 1:
        return [host_from_gpu(GpuLargeSparseTargets(node.op.what_to_output)(*node.inputs))]
    else:
        out = GpuLargeSparseTargets(node.op.what_to_output)(*node.inputs)
        return [out[0], host_from_gpu(out[1])]

optdb.register("local_large_sparse_targets_gpu", TopoOptimizer(local_optimizer([LargeSparseTargets])(local_large_sparse_targets_gpu)), 49, "fast_run")


def optimize_large_sparse_target(inputs, H, outputs, updates):
    """
    TODO: WRITEME
    """

    # need to rewrite MergeLargeSparseTargetOps because there will be multiple
    # updates containing gradH!

    if not isinstance(updates, OrderedDict):
        raise ValueError("Updates needs to be OrderedDict otherwise keys, and"
                         " values may not match after optimization")

    fgraph = gof.FunctionGraph(inputs,
                               outputs + updates.values())

    mergelst.optimize(fgraph)

    new_inputs = fgraph.inputs
    new_outputs = fgraph.outputs[:len(output)]
    new_updates = {}
    for key, update in izip(updates.keys(), fgraph.outputs[len(output):]):
        new_updates[key] = update

    return new_inputs, new_outputs, new_updates
