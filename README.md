# Factored output algorithm

Theano CPU and GPU implementations of the Large Sparse Target algorithm introduced in

[Efficient Exact Gradient Update for training Deep Networks with Very Large Sparse Targets](http://arxiv.org/abs/1412.7091),  
Pascal Vincent, Alexandre de Brébisson, Xavier Bouthillier,  
2015 NIPS.

The implementation allows to create and train any loss function from the spherical family. In the folder example, you will find an example on how to use the op to train a network with the Z-loss.

#### Following up papers:

[The Z-loss: a shift and scale invariant classification loss belonging to the Spherical Family](https://arxiv.org/abs/1604.08859),  
Alexandre de Brébisson and Pascal Vincent,  
2016.

[An Exploration of Softmax Alternatives Belonging to the Spherical Loss Family](http://arxiv.org/abs/1511.05042),  
Alexandre de Brébisson and Pascal Vincent,  
2016 ICLR.
