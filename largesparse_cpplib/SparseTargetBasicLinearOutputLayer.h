// -*- C++ -*-

// SparseTargetBasicLinearOutputLayer.h
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


#ifndef SparseTargetBasicLinearOutputLayer_INC
#define SparseTargetBasicLinearOutputLayer_INC

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>

#include <cstring>
#include <string>

//#include <blas_linalg.h>
//#include <k_sparse.h>
//#include <PP.h>
#include <SparseTargetLinearOutputLayer.h>

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
class SparseTargetBasicLinearOutputLayer: public SparseTargetLinearOutputLayer
{
public:

    // Parameter
    BMat<real> WT;    // d x D matrix WT = W^T

    // Initialization specification: WT elements will be initialized as 
    // random uniform in range (iWlow, iWhigh)
    real iWlow;
    real iWhigh;
    
    // Computed output and delta (these are computed when calling online_fbpropupdate, 
    // but are not returned by the call (to conform to the virtual prototype in base class, and because factored version doesn't compute the output)
    // They can be accessed directly as memeber variables
    BVec<real> output; // D-dimensional output
    BVec<real> delta; // D-dimensional (output - target)

    BMat<real> deltas; // minibatch of m columns of D-dimensional (output - target)


    // Default constructor
    SparseTargetBasicLinearOutputLayer()
        :SparseTargetLinearOutputLayer(), iWlow(-1), iWhigh(1)
    {}

    virtual BMat<real> get_WT() const 
    { return WT; }

    virtual void operator<<(const SparseTargetLinearOutputLayer& other)
    { WT << other.get_WT(); }

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
    SparseTargetBasicLinearOutputLayer(int d_, int D_, int K_, 
                                  real iWlow_=-1, real iWhigh_=1)
        :SparseTargetLinearOutputLayer(d_, D_, K_),
         iWlow(iWlow_), iWhigh(iWhigh_)
    {
        build();
    }



    virtual void online_fbpropupdate(                                      
        const BVec<real>& h,        // d-dimensional vector of last hidden layer
        const KSparseVec<real>& y,
        real eta, // learning_rate        
        real* L, // scalar loss
        const BVec<real>& grad_h // gradient on h (d-dimensional vector)
        )
    {
        
        bool compute_L = (L!=0);
        bool compute_grad_h = grad_h.isNotEmpty();  
        
        // check dimensions of call arguments
        assert(h.length() == d);
        assert(y.indexes.length() == K);
        assert(y.values.length() == K);
        assert( (grad_h.length() == d)  || (!compute_grad_h) );

        // Make sure temporaries are allocated to the right size
        output.resize(D);
        delta.resize(D);
        
        // output = W h = WT^T h
        blas_gemv('T', 1, WT, h, 0, output);

  
        //printf("\noutput = "); print(output);
        //printf("y.indexes = "); print(y.indexes);
        //printf("y.values = "); print(y.values);

        // delta = output - y
        dense_vec_minus_k_sparse_vec(output, y, delta);
        // printf("delta = "); print(delta);

        if (compute_L) // computing cost
        {
            // L = || delta ||^2
            *L = blas_dot(delta, delta);

            // Debug printing
            // printf(">>> Decomposing computation of L (nonfactored blas version): L = %e - 2 * %e + %e = %e\n", blas_dot(output,output), ksparse_dot(output, y.values, y.indexes), ksparse_squared_norm(y.values, y.indexes), L );
        }

        if (compute_grad_h) // computing gradient
        {
            // grad_h = 2 W^T delta = 2 WT delta
            blas_gemv('N', 2, WT, delta, 0, grad_h);
        }

        if (eta>0) // performing gradient update
        {
            blas_ger(-2*eta, h, delta, WT);
        }

    }


    virtual void batch_fbpropupdate(const BMat<real>& H,        // d x m matrix of last hidden layer activations
                                    const CKSparseMat<real>& Y, // K x m matrix 
                                    real eta, // learning_rate for parameter update
                                    // Outputs:
                                    real* L, // pointer to scalar total loss
                                    const BMat<real>& grad_H // d x m gradient on H 
        ) 
    {
        bool compute_L = (L!=0);
        bool compute_grad_H = grad_H.isNotEmpty();  
        
        // check dimensions of call arguments
        assert(H.nrows() == d);
        assert(Y.indexes.nrows() == K);
        assert(Y.values.nrows() == K);
        assert( (grad_H.nrows() == d)  || (!compute_grad_H) );

        int m = H.ncols();
        assert(Y.ncols()==m);
        assert(grad_H.ncols()==m);

        // Make sure temporaries are allocated to the right size
        deltas.resize(D,m);
        
        // outputs = W H = WT^T H
        blas_gemm_TN(1, WT, H, 0, deltas);

  
        //printf("\noutputs = "); print(deltas);
        //printf("y.indexes = "); print(y.indexes);
        //printf("y.values = "); print(y.values);

        // deltas = outputs - Y
        deltas -= Y; 
        // printf("deltas = "); print(deltas);

        if (compute_L) // computing cost
        {
            // L = || deltas ||^2
            *L = sum_prod(deltas, deltas);

            // Debug printing
            // printf(">>> Decomposing computation of L (nonfactored blas version): L = %e - 2 * %e + %e = %e\n", blas_dot(outputs,outputs), ksparse_dot(outputs, y.values, y.indexes), ksparse_squared_norm(y.values, y.indexes), L );
        }

        if (compute_grad_H) // computing gradient
        {
            // grad_H = 2 W^T deltas = 2 WT deltas
            blas_gemm_NN(2, WT, deltas, 0, grad_H);
        }

        if (eta>0) // performing gradient update
        {
            blas_gemm_NT(-2*eta, H, deltas, 1, WT);
        }

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
