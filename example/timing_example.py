import time
import numpy as np
import theano
from theano import tensor
from theano import shared
from theano.tensor.nnet import h_softmax

from utilities import create_streams

from spherical_op import GpuFactoredSphericalOp, FactoredSphericalOp


np.random.seed(42)
theano.config.floatX = 'float32'
floatX = theano.config.floatX


##########
# CONFIG #
##########
# C++ debug printing level
debug_print = 0

D = 793471  # vocabulary size
m = 200  # batch size
n_grams = 10
embedding_size = 512
invup_mode = 4  # controls how the inverse of U is updated
hasOutputBias = True  # Presence of a bias term for the output layer
learning_rate = 0.01
freq_stabilize = 100  # Frequence of the numerical stabilizations on U
hidden_sizes = [2048, 2048, 512]

# output_type controls the type of the output, it can take the following
# values:
# - lst: large sparse target algorithm
# - plain: naive gradient descent for a fully connected output
# - h_softmax: hierarchical softmax output
output_type = 'plain'  # lst, plain, h_softmax

# Loss function, either:
# - softmax: log softmax loss function. Not compatible with lst
# - square: MSE
# - z: Z-loss
cost = 'softmax' # softmax, square, z
if output_type == 'plain':
    n_batches = 10
    print_every = 1
else:
    n_batches = 100
    print_every = 10


################
# ARCHITECTURE #
################
def z_loss(AT, q, s, eps=1e-12):
    n = D  # et non pas n=hidden_sizes[-1]
    mu = s / n
    sigma = tensor.sqrt((q / n) - mu**2)
    # c = tensor.nnet.softplus((mu - AT.reshape((m,))) / (sigma + eps))
    c = tensor.nnet.softplus((mu - AT[:, 0]) / (sigma + eps))
    return c.sum()


def mse(AT, q, s):
    c = q - 2.0 * AT[:, 0] + 1.0 + 0*s
    return c.sum()

x = tensor.imatrix('ngrams')

# Embedding layer
embedding_matrix = theano.shared(
    np.random.normal(0, 0.001, (D, embedding_size)).astype(floatX))
buff = tensor.reshape(x, (-1,))
sub = embedding_matrix[buff]

# Concatenated embeddings
concat_embeddings = tensor.reshape(sub, (x.shape[0], -1))

# Hidden layers
hid = concat_embeddings
size = embedding_size * n_grams
params = []
for s in hidden_sizes:
    W = shared(np.random.normal(0, 0.01, (size, s)).astype('float32'))
    b = shared(np.zeros(s, dtype='float32'))
    params.append(W)
    params.append(b)
    hid = tensor.nnet.relu(tensor.dot(hid, W) + b)
    size = s

# Output layer
if hasOutputBias:
    hid = tensor.concatenate(
        [hid, tensor.constant(0.001 * np.ones((m, 1), dtype=floatX))], axis=1)
    d = size + 1
else:
    d = size

y_indexes = tensor.imatrix('targets')

if output_type in ['lst', 'plain']:
    V_mat = np.random.normal(0, 0.001, (D, d)).astype(floatX)
    UT_mat = np.eye(d, dtype=floatX)

    if output_type == 'lst':

        UinvT_mat = np.eye(d, dtype=floatX)
        QT_mat = np.dot(V_mat.T, V_mat)

        omega = theano.shared(np.zeros((d,), dtype=floatX))
        w_bar = theano.shared(V_mat.sum(axis=0))
        UT = theano.shared(UT_mat, name="UT")
        V = theano.shared(V_mat, name="V")
        UinvT = theano.shared(UinvT_mat)
        QT = theano.shared(QT_mat)

        if theano.config.device[:3] == 'gpu':
            factored_op = GpuFactoredSphericalOp
        elif theano.config.device[:3] == 'cpu':
            factored_op = FactoredSphericalOp
        else:
            raise ValueError

        AT, q, s, work_d, work_m = factored_op(
            eta=learning_rate,
            invup_mode=invup_mode,
            stabilize_period=freq_stabilize,
            debug_print=debug_print)(
            V, UT, UinvT, QT, omega, w_bar, hid, y_indexes)

        if cost == 'square':
            loss = mse(AT, q, s)
        elif cost == 'z':
            loss = z_loss(AT, q, s)
        else:
            raise ValueError

    elif output_type == 'plain':

        W_mat = np.dot(UT_mat, V_mat.T)
        W = theano.shared(W_mat, name="W")
        params.append(W)

        output = tensor.dot(hid, W)

        if cost == 'square':
            one_hot = tensor.zeros_like(output)
            one_hot = theano.tensor.set_subtensor(
                one_hot[theano.tensor.arange(output.shape[0]), y_indexes[:, 0]], 1.0)
            loss = ((output - one_hot) ** 2).sum(axis=1)
            loss = loss.sum()
        elif cost == 'softmax':
            y_hat = tensor.nnet.softmax(output)
            loss = tensor.nnet.categorical_crossentropy(y_hat, y_indexes[:, 0])
            loss = loss.mean()
        elif cost == 'z':
            mu = output.mean(axis=1)
            sigma = tensor.sqrt((output**2).mean(axis=1) - mu**2)
            eps = 1e-12
            oc = output[tensor.arange(output.shape[0]), y_indexes[:, 0]]
            c = tensor.nnet.softplus((mu-oc) / (sigma + eps))
            loss = c.sum()
        else:
            raise ValueError

elif output_type == 'h_softmax':

    #############
    # Config
    #############
    h_softmax_level1_size = int(np.ceil(np.sqrt(D)))
    h_softmax_level2_size = h_softmax_level1_size
    output_size = h_softmax_level1_size * h_softmax_level2_size


    #############
    # Initialize shared variables
    #############
    # First level of h_softmax
    floatX = theano.config.floatX

    # First level of h_softmax
    W1 = np.asarray(np.random.normal(0, 0.001,
                                     size=(d, h_softmax_level1_size)), dtype=floatX)
    W1 = shared(W1)
    b1 = shared(np.asarray(np.zeros((h_softmax_level1_size,)),
                           dtype=floatX))

    # Second level of h_softmax
    W2 = np.asarray(np.random.normal(0, 0.001,
                                     size=(h_softmax_level1_size, d, h_softmax_level2_size)),
                    dtype=floatX)
    W2 = shared(W2)
    b2 = shared(
        np.asarray(np.zeros((h_softmax_level1_size,
                             h_softmax_level2_size)), dtype=floatX))

    #############
    # Build graph
    #############

    # This only computes the output corresponding to the target
    y_hat_tg = h_softmax(hid, m, output_size, h_softmax_level1_size,
                         h_softmax_level2_size, W1, b1, W2, b2, y_indexes)

    # This computes all the outputs
    # output = h_softmax(hid, m, output_size, h_softmax_level1_size,
    #                       h_softmax_level2_size, W1, b1, W2, b2)

    loss = -tensor.mean(tensor.log(y_hat_tg))

    params.extend([W1, b1, W2, b2])
else:
    raise ValueError


######################
# TRAINING FUNCTIONS #
######################
grad = theano.grad(loss, [sub] + params)
grad_sub = grad[0]
grad_params = grad[1:]
update_sub = embedding_matrix, tensor.inc_subtensor(
    sub, -grad_sub * learning_rate)
update_params = [(p, p - g *learning_rate)
                 for p, g in zip(params, grad_params)]
print 'Compile function ...',
fun_train = theano.function([x, y_indexes], loss,
                            updates=[update_sub] + update_params)
print 'Done'


###########
# DATASET #
###########
train_stream, valid_stream, vocab = create_streams(m, n_grams)


############
# TRAINING #
############
it = 0
print 'Starts training...'
timing = 0
for x_mat, y_mat in train_stream.get_epoch_iterator():
    if it == n_batches:
        break
    start = time.time()
    cost = fun_train(x_mat, y_mat)
    buff = time.time() - start
    timing += buff
    if it%print_every == 0:
        print 'batch number {}:\t {}, {}'.format(it, cost, buff/m)
    it += 1

print timing / (n_batches * m)
