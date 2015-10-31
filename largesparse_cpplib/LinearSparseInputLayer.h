
// -*- C++ -*-

// LinearSparseInputLayer.h
// Copyright (C) 2014 Pascal Vincent and Universite de Montreal
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 
//  1. Redistributions of source code must retain the above copyright
//     notice, this list of conditions and the following disclaimer.
// 
//  2. Redistributions in binary form must reproduce the above copyright
//     notice, this list of conditions and the following disclaimer in the
//     documentation and/or other materials provided with the distribution.
// 
//  3. The name of the authors may not be used to endorse or promote
//     products derived from this software without specific prior written
//     permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE AUTHORS ``AS IS'' AND ANY EXPRESS OR
// IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
// OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN
// NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
// TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


#ifndef LinearSparseInputLayer_INC
#define LinearSparseInputLayer_INC

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>

#include <cstring>
#include <string>

//#include <blas_linalg.h>
#include <k_sparse.h>
#include <PP.h>

using std::string;
using std::strcmp;

// using namespace PLearn;


// #include <plearn/math/blas_proto.h>
// #include <plearn/math/lapack_proto.h>

// TODO: check we get same result when use_lower is false (and compare speed to see which alternative is faster).

// TODO: currently grad_H is set to thegradient. Maybe to more easily allow other influences we could accumulate into it 
// (to do: make it an option of the call). In that case, the user needs to clear it first before starting accumulating gradients in it.


// #define real float
#include "pl_math.h"

namespace PLearn {



// Non factored, naive version
class LinearSparseInputLayer:public PPointable
{
public:

    int D; // large dimension of sparse input
    int d; // smaller hidden layer dimension

    // Parameter
    BMat<real> WT;    // d x D column-major matrix matrix WT = W^T

    // Initialization specification: WT elements will be initialized as 
    // random uniform in range (iWlow, iWhigh)
    real iWlow;
    real iWhigh;
    
    // Default constructor
    LinearSparseInputLayer()
        :D(-1), d(-1), iWlow(-1), iWhigh(1)
    {}

    virtual BMat<real> get_WT() const 
    { return WT; }

    void initialize_params()
    {
        fill_random_uniform_continuous(WT, iWlow, iWhigh);            
    }

    void build()
    {
        if (WT.isEmpty())
        {
            WT = BMat<real>(d,D);
            initialize_params();
        }
    }

    // Convenience constructor and building
    LinearSparseInputLayer(int D_, int d_, real iWlow_, real iWhigh_)
        :D(D_), d(d_), iWlow(iWlow_), iWhigh(iWhigh_)
    {
        build();
    }


    virtual void operator<<(const LinearSparseInputLayer& other)
    {
        WT << other.WT;
    }

    virtual void fprop(const CKSparseMat<real>& input, const BMat<real>& output)
    {
        product_dense_mat_cksparse_mat(WT, input, output);
    }

    virtual void bprop_update(const CKSparseMat<real>& input, const BMat<real>& output, const BMat<real>& output_grad, real learning_rate)
    {
        if (learning_rate!=1)
            output_grad *= -learning_rate;        

        rank_update_dense_mat_cksparse_mat(output_grad, input, WT);

        if (learning_rate!=1)
            output_grad *= -1/learning_rate;
    }

//! Applys a gradient update corresponding to a weight decay on W of strength lambda, using learning rate eta
//! This is done by rescaling WT
    virtual void apply_weight_decay_update(real lambda, real eta)
    {
        real s = 1-eta*lambda;
        WT *= s;
    }

};


} // end of namespace PLearn

#endif



/*
  Local Variables:
  mode:c++
  c-basic-offset:4
  c-file-style:"stroustrup"
  c-file-offsets:((innamespace . 0)(inline-open . 0))
  indent-tabs-mode:nil
  fill-column:79
  End:
*/
// vim: filetype=cpp:expandtab:shiftwidth=4:tabstop=8:softtabstop=4:encoding=utf-8:textwidth=79 :
