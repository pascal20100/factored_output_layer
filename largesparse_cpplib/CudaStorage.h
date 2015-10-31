// -*- C++ -*-

// CudaStorage.h
//
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


#ifndef CudaStorage_INC
#define CudaStorage_INC

#include "PP.h"

#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cublas_v2.h>

namespace PLearn {

template<class T>
class CudaStorage: public PPointable
{
public:
    
    // A few STL-like typedefs
    typedef T value_type;
    typedef int size_type;
    
    // member variables
    T* dataptr_;
    int n_;     // if pitched allocation is used, this will be equal to ncols_*stride_

    // This is used for pitched allocation of column major matrices for usage through cublas
    int nrows_; 
    int ncols_;
    int stride_;  // for pitched allocation will be pitch/sizeof(T)

    inline CudaStorage()
        :dataptr_(0),
         n_(0),
         nrows_(0),
         ncols_(0),
         stride_(0)
    {}
        
    //! allocates storage for n elements of type T using cudaMalloc
    inline CudaStorage(int n)  
        :n_(n),
         nrows_(0),
         ncols_(0),
         stride_(0)         
    {
        cudaError_t cudaStat = cudaMalloc((void**)&dataptr_, n*sizeof(T));
        if (cudaStat != cudaSuccess) 
            PLERROR("device memory allocation failed");
    }

    //! allocates storage for at least ncols*nrows elements of type T using cudaMallocPitch
    inline CudaStorage( int nrows, int ncols)
        :nrows_(nrows), ncols_(ncols)
    {
        // Note that cudaMallocPitch documentation uses the semantic of row-major allocation.
        // Since we rather use column-major semantics (because we will be working with cublas),
        // for us pitch will corresponds to the offset in bytes to move to the next *column*. 
        // So the mapping of notions between this class (used for blas column major matrices) and cudaMallocPitch 
        // is width<->rows height<->columns

        size_t pitch;
        cudaError_t cudaStat = cudaMallocPitch( (void**)&dataptr_, &pitch, nrows*sizeof(T), ncols);
        if (cudaStat != cudaSuccess) 
            PLERROR("device memory allocation failed");
        if (pitch % sizeof(T)!=0)
            PLERROR("Returned pitch is not a multiple of sizeof element !");
        stride_ = pitch/sizeof(T);
        n_ = ncols_*stride_;
    }


    inline ~CudaStorage()
    {
        if (this->refcount()==0 && dataptr_!=0)
        {
            cudaFree(dataptr_);
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
