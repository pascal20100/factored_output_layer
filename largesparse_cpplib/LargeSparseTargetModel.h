// -*- C++ -*-

// LargeSparseTargetModel.h
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

#ifndef LargeSparseTargetModel_INC
#define LargeSparseTargetModel_INC

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>

#include <cstring>
#include <string>

// #define real float
#include "pl_math.h"
#include <blas_linalg.h>
#include <k_sparse.h>
#include <DataBatchStream.h>
#include <LinearSparseInputLayer.h>
#include <SparseTargetBasicLinearOutputLayer.h>
#include <SparseTargetFactoredLinearOutputLayer.h>
#include <PP.h>

namespace PLearn {

// TODO: check we get same result when use_lower is false (and compare speed to see which alternative is faster).

// TODO: currently grad_H is set to thegradient. Maybe to more easily allow other influences we could accumulate into it 
// (to do: make it an option of the call). In that case, the user needs to clear it first before starting accumulating gradients in it.

class LargeSparseTargetModel:public PPointable
{
public:
        
    // input_layer can optionally be 0 (NULL) if you want to copy the sparse input directly to the dense hidden layer
    PP<LinearSparseInputLayer> input_layer;   
    PP<SparseTargetLinearOutputLayer> output_layer;
    bool online_updates;  // whether to perform online (stochatic) update or batch updates 

    inline int d_x() const
    { 
        if (input_layer!=0)
            return input_layer->D;
        else
            return output_layer->d;
    }

    inline int d_h() const
    {
        if (input_layer!=0)
            return input_layer->d;
        else
            return 0;
    }

    inline int d_y() const
    { return output_layer->D; }

    inline int sparsity_y() const
    { return output_layer->K; }

    BMat<real> H;
    BMat<real> grad_H;

    void check_dimensions() const
    {
        if (input_layer!=0)
            assert(input_layer->d == output_layer->d);
    }

    virtual void build()
    {  
        check_dimensions();
        // No need to allocate H here. It will be resized to match minibatch size when calling methods
    }

    LargeSparseTargetModel(PP<LinearSparseInputLayer> input_layer_,
                           PP<SparseTargetLinearOutputLayer> output_layer_, bool online_updates_=false)
        : input_layer(input_layer_), output_layer(output_layer_), online_updates(online_updates_)
    {
        build();
    }

    //! copies parameters from other model, adapting them if needed so that the 2 models compute the same function
    void copy_parameters_from(const LargeSparseTargetModel& other)
    {
        if (input_layer != 0)
            *input_layer << *(other.input_layer);
        if (output_layer != 0)
            *output_layer << *(other.output_layer);

    }

    //! accumulates the loss in total_loss
    virtual void batch_fbprop_update(const CKSparseMat<real>& X, const CKSparseMat<real>& Y, 
                                     real& total_loss, real learning_rate, real eta)
    {
        assert(Y.nrows() == output_layer->D);
        if (input_layer!=0)
            assert( X.nrows() == input_layer->D );
        else
            assert( X.nrows() == output_layer->d );

        int m = X.ncols();  // minibatch size
        H.resize( output_layer->d, m);
        grad_H.resize(  output_layer->d, m);

        if (input_layer!=0)
            input_layer->fprop(X, H);
        else
            H << X;  // Simply copy sparse input X to dense H.

        real L = 0;
        output_layer->batch_fbpropupdate(                                                                                            
            H,        // d-dimensional vector of last hidden layer
            Y,
            eta, // learning_rate
            &L, // scalar loss
            grad_H // gradient on h (d-dimensional vector)
            );
        
        total_loss += L;
        
        if (input_layer!=0)
            input_layer->bprop_update(X, H, grad_H, learning_rate);
    }


    //! accumulates the loss in total_loss
    virtual void online_fbprop_update(const CKSparseMat<real>& X, const CKSparseMat<real>& Y, 
                                      real& total_loss, real learning_rate, real eta)
    {
        assert(Y.nrows() == output_layer->D);
        if (input_layer!=0)
            assert( X.nrows() == input_layer->D );
        else
            assert( X.nrows() == output_layer->d );

        int m = X.ncols(); // minibatch size
        H.resize( output_layer->d, m);
        grad_H.resize(  output_layer->d, m);

        for(int j=0; j<X.ncols(); j++)
        {
            KSparseVec<real> x = X.column(j);
            KSparseVec<real> y = Y.column(j);
            BVec<real> h = H.column(j);
            BVec<real> grad_h = grad_H.column(j);

            if (input_layer!=0)
                input_layer->fprop(x, h);
            else
                h << x;  // Simply copy sparse input x to dense h.


            real L = 0;
            output_layer->online_fbpropupdate(                                                                                            
                h,        // d-dimensional vector of last hidden layer
                y,
                eta, // learning_rate
                &L, // scalar loss
                grad_h // gradient on h (d-dimensional vector)
                );

            total_loss += L;

            if (input_layer!=0)
                input_layer->bprop_update(x, h, grad_h, learning_rate);
        }
    }

    virtual void fbprop_update(const CKSparseMat<real>& X, const CKSparseMat<real>& Y, 
                               real& total_loss, real learning_rate, real eta)
    {
        if (online_updates)
            online_fbprop_update(X, Y, total_loss, learning_rate, eta);
        else
            batch_fbprop_update(X, Y, total_loss, learning_rate, eta);
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
