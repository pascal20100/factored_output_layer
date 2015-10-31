// -*- C++ -*-

// SparseTargetLinearOutputLayer.h
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


#ifndef SparseTargetLinearOutputLayer_INC
#define SparseTargetLinearOutputLayer_INC

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>

#include <cstring>
#include <string>

#include <blas_linalg.h>
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


class SparseTargetLinearOutputLayer: public PPointable
{

public:
    int d; // last hidden layer dimension
    int D; // output layer dimension
    int K; // K-sparsity of target

    int debug_print;

protected:

    SparseTargetLinearOutputLayer(int d_=-1, int D_=-1, int K_=-1, 
                                  int debug_print_=0)
        :d(d_), D(D_), K(K_), debug_print(debug_print_)
    {}
    

public:

    virtual void build()
    {}

    virtual BMat<real> get_WT() const = 0;

    //! Copies effective parameters from other to this object
    virtual void operator<<(const SparseTargetLinearOutputLayer& other) = 0;

    virtual void online_fbpropupdate(const BVec<real>& h,        // d-dimensional vector of last hidden layer
                                     const KSparseVec<real>& y,
                                     real eta, // learning_rate
                                     
                                     // computed outputs
                                     real* L, // pointer to scalar loss
                                     const BVec<real>& grad_h // gradient on h (d-dimensional vector)
        ) = 0;

    virtual void batch_fbpropupdate(const BMat<real>& H,        // d x m matrix of last hidden layer activations
                                    const CKSparseMat<real>& Y, 
                                    real eta, // learning_rate for parameter update
                                    // Outputs:
                                    real* L, // pointer to scalar total loss
                                    const BMat<real>& grad_H // d x m gradient on H 
        ) = 0;

    virtual void apply_weight_decay_update(real lambda, real eta) = 0;
  
    virtual void print_scales() const
    {}

    virtual void consistency_check() const
    {}

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
