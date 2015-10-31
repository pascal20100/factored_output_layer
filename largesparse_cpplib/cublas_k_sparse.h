// -*- C++ -*-

// cublas_k_sparse.h
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


#ifndef cublas_k_sparse_INC
#define cublas_k_sparse_INC

#include <vector>
// #include <tuple>

#include <k_sparse.h>
#include <cublas_linalg.h>
// #include <cusparse_linalg.h>

namespace PLearn {

//! y = y + alpha x
//! with y a dense vector, and x a k-sparse vector
template<class T>
inline void blas_axpy(T alpha, const KSparseVec<T>& x, const CublasVec<T>& y)
{
    assert(x.length()==y.length());
    int K = x.K();

    for (int k=0; k<K; k++)
    {
        int i = x.indexes[k];
        if(i<0)
            break;
        T val = x.values[k];
        y.set(i, y.get(i)+alpha*val );
    }
}

//! y += x
template<class T>
inline void operator+=(const CublasVec<T>& y, const KSparseVec<T>& x)
{ blas_axpy( T(1), x, y); }

//! y -= x
template<class T>
inline void operator-=(const CublasVec<T>& y, const KSparseVec<T>& x)
{ blas_axpy( T(-1), x, y); }

// c = a-b  where a, and c are dense vectors and b is in k-sparse representation.
template<class T>
inline void dense_vec_minus_k_sparse_vec(const CublasVec<T>& a, const KSparseVec<T>& b, const CublasVec<T>& c)
{
  c << a;
  c -= b;
}

//! Y = Y + alpha X 
//! with Y a dense matrix and X a column-k-sparse matrix
template<class T>
void blas_axpy(T alpha, const CKSparseMat<T>& X, const CublasMat<T>& Y)
{
    assert(X.nrows()==Y.nrows() && X.ncols()==Y.ncols());
    int m = X.ncols();
    for (int j=0; j<m; j++)
        blas_axpy(alpha, X.column(j), Y.column(j));
}

//! Y += X
template<class T>
inline void operator+=(const CublasMat<T>& Y, const CKSparseMat<T>& X)
{ blas_axpy( T(1), X, Y); }

//! Y -= X
template<class T>
inline void operator-=(const CublasMat<T>& Y, const CKSparseMat<T>& X)
{ blas_axpy( T(-1), X, Y); }




/* y <- A x
   A is a dense m x n matrix in column major format. 
   x is a n dimensional K-sparse vector. Its K non-zero values and associated indexes are in x_values and x_indexes.
   The call will fill m-dimensional dense vector y
*/
template<class T>
void product_dense_mat_ksparse_vec(const CublasMat<T>& A, const KSparseVec<T>& x, const CublasVec<T>& y)
{
  y.clear();
  int K = x.indexes.length();
  for (int k=0; k<K; k++)
    {
      int j = x.indexes[k];
      if (j<0) // we've exhausted x (-1 indicates optional early termination, when fewer than K indexes are to be used) 
        break;
      T val = x.values[k];
      blas_axpy(val, A.column(j), y);
    }
}

/* Y <- A X
   A is a dense (m,n) matrix in column major format. 
   X is a (n,l) column-K-sparse matrix. 
   For each of its l columns, its K non-zero values and associated indexes are in the corresponding column of X_values and X_indexes (booth are (K,l) matrices ).
   The call will fill (m,l) dimensional dense matrix Y

   Multiple stream dispatch version
   
*/
template<class T>
void product_dense_mat_cksparse_mat(const CublasMat<T>& A,
                                    const CKSparseMat<T>& X,
                                    const CublasMat<T>& Y, 
                                    const CudaStreamArray& cuda_streams,
                                    const CudaEventArray& cuda_events)
{

  assert(Y.nrows()==A.nrows() && Y.ncols()==X.ncols());
  assert(A.ncols() == X.nrows());

  int l = X.ncols();

  cudaStream_t initial_stream = cublas.getStream();

  for (int j=0; j<l; j++)
    {
        cublas.setStream(cuda_streams[j%cuda_streams.length()]); // we keep the first 3 streams free for other work
        product_dense_mat_ksparse_vec(A, X.column(j), Y.column(j));
    }

  cublas.setStream(initial_stream);
}

/* Y <- A X
   A is a dense (m,n) matrix in column major format. 
   X is a (n,l) column-K-sparse matrix. 
   For each of its l columns, its K non-zero values and associated indexes are in the corresponding column of X_values and X_indexes (booth are (K,l) matrices ).
   The call will fill (m,l) dimensional dense matrix Y

   cusparse version (currently buggy TODO: debug)
*/
/* // BUGGY

template<class T>
void product_dense_mat_cksparse_mat_using_cusparse(const CublasMat<T>& A,
                                       const CKSparseMat<T>& X,
                                       const CublasMat<T>& Y, 
                                       bool synchronize_streams=true)
{    
  assert(Y.nrows()==A.nrows() && Y.ncols()==X.ncols());
  assert(A.ncols() == X.nrows());

  CusparseCSMat<T> X_csm(X);
  CublasMat<T> Y_T(Y.ncols(), Y.nrows());
  cusparse_csrmm2(1, X_csm, true, A, true, 0, Y_T);
  transpose(Y_T, Y);
}
*/

/* Y <- A X
   A is a dense (m,n) matrix in column major format. 
   X is a (n,l) column-K-sparse matrix. 
   For each of its l columns, its K non-zero values and associated indexes are in the corresponding column of X_values and X_indexes (booth are (K,l) matrices ).
   The call will fill (m,l) dimensional dense matrix Y

   gemmBatched version: this schedules many   (m,1)(1,1) -> (m,1) matrix products 
*/
template<class T>
void product_dense_mat_cksparse_mat_using_gemmBatched(const CublasMat<T>& A,
                                                      const CKSparseMat<T>& X,
                                                      const CublasMat<T>& Y)
{
    static CublasMat<float> X_values_device; // A device copy of X.values

    static CublasVec<float*> Aarray_device; // will contain pointers to the data of gemmBatched As
    static CublasVec<float*> Barray_device; // will contain pointers to the data of gemmBatched Bs
    static CublasVec<float*> Carray_device; // will contain pointers to the data of gemmBatched Cs

    static BVec<float*> Aarray_host; // will contain pointers to the data of gemmBatched As
    static BVec<float*> Barray_host; // will contain pointers to the data of gemmBatched Bs
    static BVec<float*> Carray_host; // will contain pointers to the data of gemmBatched Cs

    // The position into Aarray Barray and Carray that will be passed to the kth call to gemmBatched. 
    // Specifically call k will pass subarray starting at Aarray.data()+array_start_pos[k] 
    // and ending just before Aarray.data()+array_start_pos[k]
    static vector<int> array_start_pos; 
      
    assert(Y.nrows()==A.nrows() && Y.ncols()==X.ncols());
    assert(A.ncols() == X.nrows());

    // Maximum number of small matrix multiplies to do
    int max_n_mult = X.indexes.nrows()*X.indexes.ncols();
    Aarray_device.resize(max_n_mult, false);
    Aarray_host.resize(max_n_mult, false);
    Barray_device.resize(max_n_mult, false);
    Barray_host.resize(max_n_mult, false);
    Carray_device.resize(max_n_mult, false);
    Carray_host.resize(max_n_mult, false);

    X_values_device.resize(X.values.nrows(), X.values.ncols(), false);
    X.values >> X_values_device;

    int ncols = X.ncols();
    int K = X.K();
    array_start_pos.resize(K+1); // We'll do at most K gemmBatched launches
    array_start_pos[0] = 0;
    
    // current position (index) in the arrays Aarray, Barray and Carray to be filled 
    int current_array_pos = 0; 
        
    for (int k=0; k<K; k++)   // loop over rows of X.indexes
    {
        for(int j=0; j<ncols; j++) // loop over columns of X.indexes
        {
            // schedule  Y.column(j)  += X_values_device(k,j) A.column( X.indexes(k,j) )
            int idx = X.indexes(k,j);
            
            if(idx>=0)
            {
                // in Aarray, put address of A.column(idx)
                Aarray_host[current_array_pos] = A.data()+idx*A.stride();
                // In Barray, put address of 1x1 matrix corresponding to element X_values_device(k,j)
                Barray_host[current_array_pos] = X_values_device.data() + j*X_values_device.stride() + k;
                // In Carray put address of destination: Y.column(j)
                Carray_host[current_array_pos] = Y.data() + j*Y.stride();
                
                // Move to next position to fill
                current_array_pos++;
            }
        }
        array_start_pos[k+1] = current_array_pos;
    }

    // Now copy the host arrays to the device
    Aarray_host >> Aarray_device;
    Barray_host >> Barray_device;
    Carray_host >> Carray_device;
    
    // Finally launch the gemmBatched kernels
    for (int k=0; k<K; k++)
    {
        int start_pos = array_start_pos[k];
        int batchCount = array_start_pos[k+1]-array_start_pos[k];
        if (batchCount == 0) // we're done early
            break;

        const float alpha = 1.0f;
        const float beta = ( k==0 ? 0.0f : 1.0f ); // 

        const float** Aarray_ptr = const_cast<const float**>(Aarray_device.const_data() + start_pos);
        const float** Barray_ptr = const_cast<const float**>(Barray_device.const_data() + start_pos);
        float** Carray_ptr = Carray_device.data() + start_pos;

        int m = A.nrows();
        cublasSgemmBatched(cublas.handle(),
                           CUBLAS_OP_N,
                           CUBLAS_OP_N,
                           m, 1, 1,
                           &alpha,
                           Aarray_ptr, A.stride(),
                           Barray_ptr, 1,
                           &beta,
                           Carray_ptr, Y.stride(), 
                           batchCount);
    }

}


// OK so for some reason std::tuple is not currently usable in this install (apparently uses new C++ experimental features)
// so I'll write my own silly class to hold 3 ints
class TripleInt
{
public:
    int i,j,k;
    
    inline TripleInt(int i_=0, int j_=0, int k_=0)
        :i(i_), j(j_), k(k_)
    {}    
};

//! A <- A + U V^T  Does a rank l update to matrix A.
//! A is a dense (m,n) matrix in column major representation
//! U is a dense (m,l) matrix in column major representation
//! V is a sparse (n,l) matrix in column-K-sparse representation
//!  gemmBatched version: this schedules many   (m,1)(1,1) -> (m,1) matrix products 


template<class T>
void rank_update_dense_mat_cksparse_mat_using_gemmBatched(const CublasMat<T>& U,
                                                          const CKSparseMat<T>& V, 
                                                          const CublasMat<T>& A)
{
    
    // cudaDeviceSynchronize();


    static CublasMat<float> V_values_device; // A device copy of V.values

    static CublasVec<float*> Aarray_device; // will contain pointers to the data of gemmBatched As
    static CublasVec<float*> Barray_device; // will contain pointers to the data of gemmBatched Bs
    static CublasVec<float*> Carray_device; // will contain pointers to the data of gemmBatched Cs

    static BVec<float*> Aarray_host; // will contain pointers to the data of gemmBatched As
    static BVec<float*> Barray_host; // will contain pointers to the data of gemmBatched Bs
    static BVec<float*> Carray_host; // will contain pointers to the data of gemmBatched Cs

    // Note: important for understanding this code: A (third destination matrix of this call) 
    // has nothing to do with Aarray (which will contain first matrix for blas gemm call)

    assert( A.nrows() == U.nrows() && A.ncols()==V.nrows() && U.ncols() == V.ncols() );    

    // Maximum number of small matrix multiplies to do
    int max_n_mult = V.indexes.nrows()*V.indexes.ncols();
    Aarray_device.resize(max_n_mult, false);
    Aarray_host.resize(max_n_mult, false);
    Barray_device.resize(max_n_mult, false);
    Barray_host.resize(max_n_mult, false);
    Carray_device.resize(max_n_mult, false);
    Carray_host.resize(max_n_mult, false);
    V_values_device.resize(V.values.nrows(), V.values.ncols(), false);
    V.values >> V_values_device;

    int ncols = V.ncols();
    int K = V.K();


    // This vector will contain TripleInt containing i,j,k where k,j are row and column indexes in V.indexes
    // and i is the associated row index of sparse matrix V. i.e. we have V.indexes(k,j)=i
    static std::vector< TripleInt > unscheduled;

    // will contain the indexes of the columns already scheduled for this round
    static std::vector< int > columns_touched_this_round; 

    // This is a boolean vector of length A.ncols() that we use to detect whether a 
    // specific column has already been incorporated into the current scheduled launch 
    static std::vector< bool > column_is_touched;
    column_is_touched.resize(A.ncols() ,false);

    int nnz = V.indexes.nrows()*V.indexes.ncols();;
    unscheduled.reserve(nnz);
    unscheduled.clear();

    columns_touched_this_round.reserve(nnz);    
    columns_touched_this_round.clear();

    // First populate unscheduled from V.indexes
    for(int j=0; j<ncols; j++) // loop over columns of V.indexes
    {
        for (int k=0; k<K; k++)   // loop over rows of V.indexes
        {
            int i = V.indexes(k,j);
            if (i<0)
                break;
            unscheduled.push_back(TripleInt(i,j,k));
        }
    }

    // launch_start_pos[k] contains the position into scheduled, Aarray, Barray, and Carray that will be passed to the kth call to gemmBatched. 
    // Specifically launch k will pass subarray starting at Aarray.data()+launch_start_pos[k] 
    // and ending just before Aarray.data()+launch_start_pos[k]
    static vector<int> launch_start_pos; 
    launch_start_pos.clear();
    launch_start_pos.push_back(0); // first launch start position

    // current position (index) in the arrays Aarray, Barray and Carray to be filled 
    int scheduled_array_pos = 0; 

    // Now move things from unscheduled to scheduled
    while( !unscheduled.empty() )  // loop over planned launches
    {
        int old_n_unscheduled = unscheduled.size();
        int new_n_unscheduled = 0;
        for (int p = 0; p<old_n_unscheduled; p++)
        {
            TripleInt trip = unscheduled[p];
            // non-zero element in i_th row of sparse matrix V implies product will accumulate in i_th column of A
            if ( column_is_touched[trip.i] )  // we already scheduled that column for update in this round
            {
                unscheduled[new_n_unscheduled++] = trip; // so we register it back into unscheduled
            }
            else  // register it as touched and schedule it
            {
                columns_touched_this_round.push_back(trip.i);
                column_is_touched[trip.i] = true; // remember column i has been scheduled for this launch so we don't add it a second time

                // Now schedule the gemm operation

                // we want to do A.column(i) += V(i,j) U.column(j)
                // i.e.          A.column(i) += V.values(k,j) U.column(j)
                // i.e.          A.column(i) +=   U.column(j)   V.values(k,j) 
                //               -----------    ----------     ----------
                // dim,putinto:  (m,1)->Carray  (m,1)->Aarray   (1,1)->Barray

                // in Aarray, put address of U.column(j)
                Aarray_host[scheduled_array_pos] = U.data()+trip.j*U.stride();
                // In Barray, put address of 1x1 matrix corresponding to element V_values_device(k,j)
                Barray_host[scheduled_array_pos] = V_values_device.data() + trip.j*V_values_device.stride() + trip.k;
                // In Carray put address of destination: A.column(i)
                Carray_host[scheduled_array_pos] = A.data() + trip.i*A.stride();

                ++scheduled_array_pos;                
            }
        }
        // we're done scheduling this round
        unscheduled.resize(new_n_unscheduled);

        // erase the column_is_touched bits we just set (associated to things we moved in scheduled)
        // so that it will be in a clear state to for next planned launch
        for (unsigned int pos = 0; pos<columns_touched_this_round.size(); pos++)
            column_is_touched[ columns_touched_this_round[pos] ] = false;  // clear all previously set bits
        columns_touched_this_round.clear();

        // register the current scheduled_array_pos
        launch_start_pos.push_back(scheduled_array_pos);

        // we've completed the schedule for this launch, now redo this for next
    }

    // Now we have a properly ordered schedule, 

    // Now copy the host arrays to the device
    Aarray_host >> Aarray_device;
    Barray_host >> Barray_device;
    Carray_host >> Carray_device;
    
    // Finally launch the gemmBatched kernels
    for (unsigned int k=0; k<launch_start_pos.size()-1; k++)
    {
        int start_pos = launch_start_pos[k];
        int batchCount = launch_start_pos[k+1]-launch_start_pos[k];
        if (batchCount == 0) // we're done early
            break;

        const float alpha = 1.0f;
        const float beta = 1.0f; 

        const float** Aarray_ptr = const_cast<const float**>(Aarray_device.const_data() + start_pos);
        const float** Barray_ptr = const_cast<const float**>(Barray_device.const_data() + start_pos);
        float** Carray_ptr = Carray_device.data() + start_pos;

        int m = A.nrows();
        cublasSgemmBatched(cublas.handle(),
                           CUBLAS_OP_N,
                           CUBLAS_OP_N,
                           m, 1, 1,
                           &alpha,
                           Aarray_ptr, U.stride(),
                           Barray_ptr, 1,
                           &beta,
                           Carray_ptr, A.stride(), 
                           batchCount);
    }

    // cudaDeviceSynchronize();

}



/* Y <- A X
   A is a dense (m,n) matrix in column major format. 
   X is a (n,l) column-K-sparse matrix. 
   For each of its l columns, its K non-zero values and associated indexes are in the corresponding column of X_values and X_indexes (booth are (K,l) matrices ).
   The call will fill (m,l) dimensional dense matrix Y
*/
//! Naive sequential version not explicitly using streams
template<class T>
void product_dense_mat_cksparse_mat(const CublasMat<T>& A,
                                    const CKSparseMat<T>& X,
                                    const CublasMat<T>& Y)
{
  assert(Y.nrows()==A.nrows() && Y.ncols()==X.ncols());
  assert(A.ncols() == X.nrows());

  int l = X.ncols();
  for (int j=0; j<l; j++)
      product_dense_mat_ksparse_vec(A, X.column(j), Y.column(j));

}

// Copies b to a 
template<class T> 
inline void operator<<(const CublasVec<T>& a, const KSparseVec<T>& b)
{
    assert(a.length() == b.length());
    a.clear();
    for (int k=0; k<b.indexes.length(); k++)
    {
        int i = b.indexes[k];
        if (i<0)
            break;
        a.set(i, b.values[k]);
    }
}


// Copies a to b
template<class T> 
inline void operator>>(const CublasVec<T>& a, const KSparseVec<T>& b)
{ operator<<(b,a); }

// Copies B to A
template<class T> 
inline void operator<<(const CublasMat<T>& A, const CKSparseMat<T>& B)
{
    assert(A.nrows() == B.nrows() && A.ncols() == B.ncols() );
    A.clear();
    for (int j=0; j<A.ncols(); j++)
        A.column(j) << B.column(j);
}

// Copies A to B
template<class T> 
inline void operator>>(const CublasMat<T>& A, const CKSparseMat<T>& B)
{ operator<<(B,A); }




// A <- A + u v^T
// A is a dense (m,n) matrix in column major representation
// u is a dense m-dimensional vector
// v is a sparse n-dimensional vector in K-sparse representation
template<class T>
void rank_one_update_dense_vec_ksparse_vec(const CublasVec<T>& u, const KSparseVec<T>& v, const CublasMat<T>& A)
{
  int K = v.K();
  for(int k=0; k<K; k++)
    {
        int j = v.indexes[k];
        if (j<0) // we've exhausted v (-1 indicates optional early termination, when fewer than K indexes are to be used) 
            break;
        blas_axpy(v.values[k], u, A.column(j) );
    }
}



//! A <- A + U V^T  Does a rank l update to matrix A.
//! A is a dense (m,n) matrix in column major representation
//! U is a dense (m,l) matrix in column major representation
//! V is a sparse (n,l) matrix in column-K-sparse representation

//! This is the streamed version: 
//! The call is non-blocking for the host, and will return as soon as everything is scheduled.
//! - cuda_streams is a CudaStreamArray vector of streams that will be used to spawn the many computaions
//! - cuda_events must be a CudaEventArray with at least one more event than there are streams in the CudaStreamArray
//! - parameters_ready_sream : completion on which the call will initially synchronize (wait for completion)
//!     so when parameters_ready_sream is done, U,V, and A must be ready for use.
//!     Default value of 0 means use the "current" stream (given by cublas.getStream(), not the "default" stream ( unless the current is the "default" ) 
//! - computation_done_stream : is the stream on which all completion events of this method will be recorded 
//!     so the user can synchronize on computation_done_stream when he wants to ensure the operation is completed
//!     or can register subsequent operations (that need the result) on that stream.
//!     Default value of 0 means use the "current" stream (given by cublas.getStream(), not the "default" stream ( unless the current is the "default" ) 

template<class T>
void rank_update_dense_mat_cksparse_mat(const CublasMat<T>& U,
                                        const CKSparseMat<T>& V, 
                                        const CublasMat<T>& A,
                                        const CudaStreamArray& cuda_streams,
                                        const CudaEventArray& cuda_events,
                                        cudaStream_t parameters_ready_stream = 0, 
                                        cudaStream_t computation_done_stream = 0)
{
    assert( A.nrows() == U.nrows() && A.ncols()==V.nrows() && U.ncols() == V.ncols() );

    cudaStream_t cublas_orig_stream = cublas.getStream();
    if (parameters_ready_stream == 0)
        parameters_ready_stream = cublas_orig_stream;
    if ( computation_done_stream == 0)
        computation_done_stream = cublas_orig_stream;


    static BVec<int> n_scheduled;
    n_scheduled.resize(cuda_streams.length());
    n_scheduled.clear();

    cudaEvent_t start_event = cuda_events[0];    
    cudaEventRecord(start_event, parameters_ready_stream);

    // cudaDeviceSynchronize();

  // loop over rank-one updates
  int l = U.ncols();
  
  for(int j=0; j<l; j++)
    {
      CublasVec<T> u = U.column(j);
      KSparseVec<T> v = V.column(j);
      int K = v.K();
      for(int k=0; k<K; k++)
      {
          int j = v.indexes[k];
          if (j<0) // we've exhausted v (-1 indicates optional early termination, when fewer than K indexes are to be used) 
              break;
          int streamnum = j%cuda_streams.length();
          cublas.setStream(cuda_streams[streamnum]);
          if (n_scheduled[streamnum]==0) // first scheduling in this stream
              cudaStreamWaitEvent(cuda_streams[streamnum], start_event, 0);  // make it wait for start_event
          blas_axpy(v.values[k], u, A.column(j) );
          n_scheduled[streamnum]++;
      }
    }

  for(int k=0; k<cuda_streams.length(); k++)
      {
          if (n_scheduled[k] > 0)
          {
              cudaEvent_t done_stream_k_event = cuda_events[1+k]; // we've already used the first for the start_event    
              cudaEventRecord(done_stream_k_event, cuda_streams[k]);
              cudaStreamWaitEvent(computation_done_stream, done_stream_k_event, 0);
          }
      }
  
  // cudaDeviceSynchronize();
  cublas.setStream(cublas_orig_stream);

}



// Naive IMPLEMNTATION NOT USING STREAMs
template<class T>
void rank_update_dense_mat_cksparse_mat(const CublasMat<T>& U,
                                        const CKSparseMat<T>& V, 
                                        const CublasMat<T>& A)
{
    assert( A.nrows() == U.nrows() && A.ncols()==V.nrows() && U.ncols() == V.ncols() );

  // loop over rank-one updates
  int l = U.ncols();

  for(int j=0; j<l; j++)
    {
      rank_one_update_dense_vec_ksparse_vec(U.column(j), V.column(j), A);
    }
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
