// -*- C++ -*-

// CpuStorage.h
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


#ifndef CpuStorage_INC
#define CpuStorage_INC

#include "PP.h"


#ifdef DO_NOT_INCLUDE_CUDA
#define DEFAULT_MEMTYPE CPPNEW
#else
#include <cuda_runtime_api.h>
#include <cuda.h>
#define DEFAULT_MEMTYPE CUDA_HOST
#endif

#define DEFAULT_MEMTYPE CUDA_HOST


namespace PLearn {

enum MemoryType { CPPNEW, CUDA_HOST };

template<class T>
class CpuStorage: public PPointable
{
public:
// A few STL-like typedefs
    typedef T value_type;
    typedef int size_type;
    typedef T* iterator;
    typedef const T* const_iterator;

    // member variables
    T* dataptr_;
    MemoryType memtype_;
    int n_;

    inline CpuStorage()
        :dataptr_(0),
         memtype_(CUDA_HOST),
         n_(0)
    {}
        
    inline CpuStorage( int n, MemoryType memtype = CUDA_HOST)
        :memtype_(memtype), n_(n)
    {
        switch(memtype_)
        {
        case CPPNEW:
            // fprintf(stderr,"######### Using C++ new  ##########\n");
            dataptr_ = new T[n];
            break;
        case CUDA_HOST:
#ifdef DO_NOT_INCLUDE_CUDA
            // fprintf(stderr,"######### Using new instead of cudaMallocHost ##########\n");
            dataptr_ = new T[n];
#else
            // fprintf(stderr,"######### Using cudaMallocHost ##########\n");
            cudaError_t err = cudaMallocHost( (void**)&dataptr_, n*sizeof(T)); 
            if (err!=cudaSuccess)
                PLERROR("Failed to allocate memory on host");
#endif
            break;
        }
    }

    inline ~CpuStorage()
    {
        if (this->refcount()==0 && dataptr_!=0)
        {
            switch(memtype_)
            {
            case CPPNEW:
                delete[] dataptr_;
                // printf("FREED %d (size %d)\n", n_, (int)sizeof(T)); 
                break;
            case CUDA_HOST:
#ifdef DO_NOT_INCLUDE_CUDA
                delete[] dataptr_;
#else
                cudaFreeHost(dataptr_); 
#endif
                break;
            }
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
