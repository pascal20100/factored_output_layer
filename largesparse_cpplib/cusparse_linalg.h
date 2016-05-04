// -*- C++ -*-

// cusparse_linalg.h
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


#ifndef cusparse_linalg_INC
#define cusparse_linalg_INC

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <math.h>



// #define real float
#include "pl_math.h"

#include "plerror.h"

#include "cusparse_v2.h"

#include "cublas_linalg.h"
#include "k_sparse.h"

namespace PLearn {

using std::max;
using std::min;

class CusparseInit
{
private:
    cusparseHandle_t handle_;
    
public:

    inline cusparseHandle_t handle() const
    {
        return handle_;
    }
    
    //! utility function to check the return status of a cusparse call
    inline void check_status(cusparseStatus_t cusparse_stat) const
    { 
        if (cusparse_stat != CUSPARSE_STATUS_SUCCESS) 
            PLERROR("CUSPARSE operation failed");
    }

    inline void setStream(cudaStream_t streamId) const
    { check_status( cusparseSetStream(handle(), streamId) ); }

    // used cuda device num will be looked up in environment variable CUDA_DEVICE_NUM
    CusparseInit()
    {
        cusparseStatus_t cusparse_stat;      
        cusparse_stat = cusparseCreate(&handle_); 

        if (cusparse_stat != CUSPARSE_STATUS_SUCCESS) 
        { 
            printf ("CUSPARSE initialization failed\n   cusparseStatus_t = %d\n", cusparse_stat); 
        }
        else
            printf("CusparseInit() successful!\n");
        
        cusparsePointerMode_t mode;
        cusparse_stat = cusparseGetPointerMode(handle_, &mode);
        if (cusparse_stat != CUSPARSE_STATUS_SUCCESS) 
        { printf ("CUSPARSE failed to get initial pointer mode\n"); }        

    }

    inline void setPointerModeDevice() const
    { cusparseSetPointerMode(handle(), CUSPARSE_POINTER_MODE_DEVICE); }

    inline void setPointerModeHost() const
    { cusparseSetPointerMode(handle(), CUSPARSE_POINTER_MODE_HOST); }


    ~CusparseInit()
    {
        cusparseDestroy(handle());
    }
};

static CusparseInit cusparse;

// *************************
// *   CusparseCSMat class
// *************************
// This CusparseCSMat class is meant to be but a thin wrapper around a column-major cusparse matrix residing in device memory

template<class T>
class CusparseCSMat {

protected:

    bool transposed_;   // if false represents CSR if true CSC
    int nrows_;
    int ncols_; 

public:
    CublasVec<T> csrValA;
    CublasVec<int> csrRowPtrA;
    CublasVec<int> csrColIndA;

    cusparseMatDescr_t descrA;

protected:

    void createMatDescr()
    {
        cusparse.check_status( cusparseCreateMatDescr(&descrA) );
        cusparseSetMatType(descrA,CUSPARSE_MATRIX_TYPE_GENERAL);
        cusparseSetMatIndexBase(descrA,CUSPARSE_INDEX_BASE_ZERO);
        cusparseSetMatFillMode(descrA, CUSPARSE_FILL_MODE_LOWER);
        cusparseSetMatDiagType(descrA, CUSPARSE_DIAG_TYPE_NON_UNIT);        
    }

    void destroyMatDescr()
    {
        cusparse.check_status( cusparseDestroyMatDescr(descrA) );
    }

    //! will return a vector usable as a csrRowPtr 
    //! built for a k-sparse matrix
    static BVec<int> get_row_ptr_vec_for_cksparse(int n, int k) 
    {
        static BVec<int> row_ptrs;
        row_ptrs.resize(n+1);
        int* row_ptrs_data = row_ptrs.data();
        int i, pos;
        for(i=0, pos=0; i<n; i++, pos+=k)
        {
            row_ptrs_data[i] = pos;
        }
        row_ptrs[i] = pos;
        
        return row_ptrs;
    }


public:
    inline bool transposed() const { return transposed_; }
    inline int nrows() const { return (transposed_ ?ncols_ :nrows_); }
    inline int ncols() const { return (transposed_ ?nrows_ :ncols_); }

    //! returns the number of non-zero elements.
    inline int nnz() const { return csrValA.length(); }

    CusparseCSMat(int nrows, int ncols, int nnz, bool transp=false)
        :transposed_(transp), nrows_(nrows), ncols_(ncols), 
         csrValA(nnz), 
         csrRowPtrA((transp ?ncols :nrows)+1), 
         csrColIndA(nnz)
    {
        createMatDescr();
    }

    
    void operator<<(const CKSparseMat<T> A) const
    {
        if (!transposed_ ||
            nrows_ != A.ncols() ||
            ncols_ != A.nrows() || 
            A.K()*A.ncols() != nnz() )
            PLERROR("Sparse matrices have incompatible size for being copied with operator<<");

        csrValA << A.values.flat();
        csrColIndA << A.indexes.flat();
        csrRowPtrA << get_row_ptr_vec_for_cksparse( A.ncols() , A.K() );
    }

    CusparseCSMat(const CKSparseMat<T>& A)
        :transposed_(true), 
         nrows_(A.ncols()), ncols_(A.nrows()),
         csrValA( A.ncols()*A.K() ), 
         csrRowPtrA( A.ncols()+1 ), 
         csrColIndA( A.ncols()*A.K() )
    {
        operator<<(A);
    }

    ~CusparseCSMat()
    {
        destroyMatDescr();
    }


};


// C <- alpha A B + beta C
// with optionally trsnposed of A and/or of B  (but not of C)
inline void cusparse_csrmm2(float alpha, const CusparseCSMat<float>& A, bool transA, const CublasMat<float>& B, bool transB, float beta, const CublasMat<float>& C)
{
    assert( A.nrows()==C.nrows() );
    assert( A.ncols()==B.nrows() );
    assert( B.ncols()==C.ncols() );

    int m = A.nrows();
    int n = B.ncols();
    int k = A.ncols();
    int nnz = A.nnz();

    if( A.transposed() )
        transA = !transA;

    cusparse.setPointerModeHost();
    cusparse.check_status( cusparseScsrmm2(cusparse.handle(), 
                                           (transA ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE), 
                                           (transB ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE), 
                                           m, 
                                           n, 
                                           k, 
                                           nnz, 
                                           &alpha, 
                                           A.descrA, 
                                           A.csrValA.const_data(), 
                                           A.csrRowPtrA.const_data(), 
                                           A.csrColIndA.const_data(),
                                           B.const_data(), 
                                           B.stride(),
                                           &beta, 
                                           C.data(), 
                                           C.stride() )  );
    
}



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
