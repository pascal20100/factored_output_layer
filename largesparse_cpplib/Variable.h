// -*- C++ -*-

// Variable.h
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


#ifndef Variable_INC
#define Variable_INC

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

#include "pl_math.h"

namespace PLearn {

//! To DO:
//! make all matrix-like things derive from abstract base class GMat which itself derives from Object
//! Then define basic blas and similar operations on GMat that call the appropriate specialized versions

class GMat: public Object
{
    enum MatType { UNSPECIFIED, BLASMAT_FLOAT, BLASMAT_DOUBLE, CUBLASMAT_FLOAT, CUBLASMAT_DOUBLE, CKSPARSEMAT_FLOAT, CKSPARSEMAT_DOUBLE};

public:
    MatType mat_type;
};


void gemm_NN(double alpha, double beta, GMat* A, GMat* B, GMat* C)  
{
    if ( A->mat_type == B->mat_type && A->mat_type == C->mat_type )
    {
        switch( C->mat_type )
        {
        case BLASMAT_FLOAT:
            gemm_NN(alpha, beta, *(BlasMat<float>*)A, *(BlasMat<float>*)B, *(BlasMat<float>*)C);
            break;
        case BLASMAT_DOUBLE:
            break;            
        }
    }
    else
    {
        PLERROR("Heterogenous mat types not yet implemented");
    }
}

class Variable: public Object
{
    
public:

    enum GradUpdateMode {DONT_BPROP_HERE, SET_GRAD, ACCUMULATE_GRAD, ACCUMULATE_GRAD_AND_UPDATE_VAL, UPDATE_VAL_DIRECTLY};

    std::vector< PP<Variable> > parents;
    NodeRolde node_role;

    int n_children;
    bool update_me; // indicates whether this node is a parameter whose value we need to update
    bool dont_bprop_here;

    // Must be set to 0 before every bprop (fprop sets it to 0)
    int n_gradient_contribs_received;

    //! Returns the appropriate update mode for this parent
    inline GradUpdateMode getGradientUpdateModeForParent(Variable* parent)
    {
        GradUpadeMode update_mode = DONT_BPROP_HERE;
        if (!parent->dont_bprop_here)
        {
            if (parent->n_children==1) // we're the only child
            {
                if (parent->update_me) // parent is a parameter we want to update
                    update_mode = UPDATE_VAL_DIRECTLY;
                else // parent is an intermediate variable, directly set its gradient
                    update_mode = SET_GRAD;
            }
            else // there's more than one child
            { 
                parent->n_gradient_contribs_received += 1;
                if (parent->update_me && parent->n_gradient_contribs_received == parent->n_children) // this is the last contributrion to a parameter gradient
                    update_mode = ACCUMULATE_GRAD_AND_UPDATE_VAL;
                else 
                    update_mode = ACCUMULATE_GRAD;
            }
        }
        return update_mode;
    }

    PP<Object> value;
    PP<Object> gradient;
    PP<Object> learning_rates; // may amount to a 

    virtual void sizeprop() = 0;

    //! computes value based on parents' values
    virtual void fprop()
    {
        n_gradient_contribs_received = 0;
    }

    virtual void update(real learning_rate)
    {
        blas_axpy(learning_rate, value), dynamic_cast<Mat*>(value), 
            
    }

    virtual void bprop(real learning_rate)
    {
        Object* parent0 = parents[0];
        GMat* parent0_value_gmat = dynamic_cast< GMat* >(parent0->value);

        if (parent0_blasmat!=0)
        {
            switch( getGradientUpdateModeForParent(parent0) )
            {
            case DONT_BPROP_HERE:
                break;
            case SET_GRAD:
                break;
            case ACCUMULATE_GRAD:
                break;
            case ACCUMULATE_GRAD_AND_UPDATE_VAL: 
                break;
            case UPDATE_VAL_DIRECTLY:
                break;
            }
        }
    }
    
};

template<class ValueType>
class Variable: public AbstractVariable
{
    
public:

    typedef ValueType value_type;

    ValueType value;
    ValueType gradient;

};

template<class ValueType>
class InputVariable: public Variable<ValueType>
{
public:
    
    InputVariable()
    {}

    InputVariable(const ValueType& val, const ValueType& grad)
        :value(val), gradient(grad)
    {}    
    
};


template<class ValueType>
class MatrixProductVariable: public Variable<ValueType>
{
public:
    
    MatrixProductVariable(PP<Variable> weights, PP<Variable> input)
    {
        parents.push_back(weights);
        parents.push_back(input);
    }    

    virtual void fprop()
    {
        AbstractVariable* weights = parents[0];
        AbstractVariable* input = parents[1];
        
        Variable<CublasMat<double> >* cublas_weights = dynamic_cast< Variable<CublasMat<double> > >(weights);
        Variable<CublasMat<double> >* cublas_input = dynamic_cast< Variable<CublasMat<double> > >(input);
        if ( cublas_input != 0 && cublas_weights!=0 )
        {
            value.resize( cublas_weights->value.nrows(), cublas_input->value.ncols() );
            blas_gemm_NN( cublas_weight->value, cublas_input->value, value );
        }
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
