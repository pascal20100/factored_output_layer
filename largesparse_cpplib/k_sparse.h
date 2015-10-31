// -*- C++ -*-

// k_sparse.h
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


#ifndef k_sparse_INC
#define k_sparse_INC

#include <blas_linalg.h>

namespace PLearn {


// prefetch things are copied and adapted from Linux/include/linux/prefetch.h
// should try to include that or otherwise redefine it here

// need to find out the optimal stride for thearchitecture
#define PREFETCH_STRIDE 8
// this prefetch is probably gcc specific need to gracefully handleother compilers
#define prefetch(x) __builtin_prefetch(x)

inline void prefetch_array(char *data, int len)
{
  char *end = data + len;
  
  for (char* cp = (char*)data; cp < end; cp += PREFETCH_STRIDE)
    prefetch(cp);
}

template<class T>
inline void prefetch_array(T *data, int len)
{ prefetch_array( (char*)data, len*sizeof(T) ); }


template<class Telem> 
class KSparseVec
{
public:
    int length_;         // length of sparse vector
    BVec<Telem> values;  
    BVec<int> indexes;  

    inline int length() const
    { return length_; }

    //! Returns sparsity K
    inline int K() const
    { return indexes.length(); }


    KSparseVec()
        :length_(0)
    {}

    KSparseVec(int d, int K)
        :length_(d),
         values(K)
    {}

    KSparseVec(int d, const BVec<Telem>& values_, const BVec<int>& indexes_)
        :length_(d), values(values_), indexes(indexes_)
    {}

};


//! A class of m x n  column-k-sparse matrix

template<class Telem> 
class CKSparseMat
{
public:

    int nrows_; // dimension of columns m
    BMat<Telem> values;  // K x n
    BMat<int> indexes;  // K x n

    inline int nrows() const 
    { return nrows_; }
    
    inline int ncols() const
    { return indexes.ncols(); }

    //! Returns sparsity K
    inline int K() const
    { return indexes.nrows(); }

    CKSparseMat()
        :nrows_(0)
    {}

    //! Allocates a m x n column-K-sparse matrix (sotred as a K x n matrix of 
    //! integer indexes and another K x n matrix of associated real values)
    CKSparseMat(int m, int n, int K)
        :nrows_(m), 
         values(K,n),
         indexes(K,n)
    {}

    CKSparseMat(int m, const BMat<Telem>& values_, const BMat<int>& indexes_)
        :nrows_(m), values(values_), indexes(indexes_)
    {}

    //! Constructs the view of the vector passed as argument as a single column sparse matrix
    CKSparseMat(const KSparseVec<Telem>& x)
        :nrows_(x.length() ), values(x.values), indexes(x.indexes)
    {}

        
    inline KSparseVec<Telem> column(int j) const
    {
        return KSparseVec<Telem>(nrows_, values.column(j), indexes.column(j));
    }
    

    //! Slow op, but used for printing...
    inline Telem operator()(int i, int j) const
    {
        BVec<int> idx_j = indexes.column(j);
        int K = idx_j.length();
        for (int k=0; k<K; k++)
        {
            int ik = idx_j[k];
            if (ik==i)
                return values(k,j);
            if (ik<0 || ik>i)
                break;
        }
        return Telem(0);
    }

};


  template<class T> 
  void print(const CKSparseMat<T>& A)
  {
      printf("CKSparseMat %d x %d  (column K-sparsity =%d)\n", A.nrows(), A.ncols(), A.K());
      for (int i=0; i<A.nrows(); i++)
      {
          for (int j=0; j<A.ncols(); j++)
          {
            print( A(i,j) );
            printf(" ");
          }
        printf("\n");
      }
  }


// Copies b to a 
template<class T> 
inline void operator<<(const BVec<T>& a, const KSparseVec<T>& b)
{
    assert(a.length() == b.length());
    a.clear();
    for (int k=0; k<b.indexes.length(); k++)
    {
        int i = b.indexes[k];
        if (i<0)
            break;
        a[ i ] = b.values[k];
    }
}

// Copies a to b
template<class T> 
inline void operator>>(const BVec<T>& a, const KSparseVec<T>& b)
{ operator<<(b,a); }

// Copies B to A
template<class T> 
inline void operator<<(const BMat<T>& A, const CKSparseMat<T>& B)
{
    assert(A.nrows() == B.nrows() && A.ncols() == B.ncols() );
    A.clear();
    for (int j=0; j<A.ncols(); j++)
        A.column(j) << B.column(j);
}

// Copies A to B
template<class T> 
inline void operator>>(const BMat<T>& A, const CKSparseMat<T>& B)
{ operator<<(B,A); }


//! y = y + alpha x
//! with y a dense vector, and x a k-sparse vector
template<class T>
inline void blas_axpy(T alpha, const KSparseVec<T>& x, const BVec<T>& y)
{
    assert(x.length()==y.length());
    int K = x.K();

    for (int k=0; k<K; k++)
    {
        int i = x.indexes[k];
        if(i<0)
            break;
        T val = x.values[k];
        y[i] += alpha*val;
    }
}

//! y += x
template<class T>
inline void operator+=(const BVec<T>& y, const KSparseVec<T>& x)
{ blas_axpy( T(1), x, y); }

//! y -= x
template<class T>
inline void operator-=(const BVec<T>& y, const KSparseVec<T>& x)
{ blas_axpy( T(-1), x, y); }

// c = a-b  where a, and c are dense vectors and b is in k-sparse representation.
template<class T>
inline void dense_vec_minus_k_sparse_vec(const BVec<T>& a, const KSparseVec<T>& b, const BVec<T>& c)
{
  c << a;
  c -= b;
}


//! Y = Y + alpha X 
//! with Y a dense matrix and X a column-k-sparse matrix
template<class T>
void blas_axpy(T alpha, const CKSparseMat<T>& X, const BMat<T>& Y)
{
    assert(X.nrows()==Y.nrows() && X.ncols()==Y.ncols());
    int m = X.ncols();
    for (int j=0; j<m; j++)
        blas_axpy(alpha, X.column(j), Y.column(j));
}

//! Y += X
template<class T>
inline void operator+=(const BMat<T>& Y, const CKSparseMat<T>& X)
{ blas_axpy( T(1), X, Y); }

//! Y -= X
template<class T>
inline void operator-=(const BMat<T>& Y, const CKSparseMat<T>& X)
{ blas_axpy( T(-1), X, Y); }

//! Computed the squared eurlidean norm of x given in k-sparse representation
template<class T>
T ksparse_squared_norm(const BVec<T>& x_values, const BVec<int>& x_indexes)
{
    assert(x_values.length() == x_indexes.length());
  T sqn = 0;
  for(int k=0; k<x_indexes.length(); k++)
    {
      if (x_indexes[k]<0) // we've exhausted x
        break;
      T val = x_values[k];
      sqn += val*val;
    }
  return sqn;
}


//! Computes the dot product between a dense vector u and a k-sparse vector v
//! According to k-sparse representation, the indexes must be in ascending order, 
//! except for an optional terminal -1 that can be used to indicate that fewer than the length of the index vector are used.
template<class T>
T ksparse_dot(const BVec<T>& u, const BVec<T>& v_values, const BVec<int>& v_indexes)
{
    assert(v_values.length() == v_indexes.length());
  T dp = 0;  
  for( int k=0; k<v_indexes.length(); k++ )
    {
      int i = v_indexes[k];
      if (i<0)
        break;
      T v_val = v_values[k];
      dp += u[i] * v_val;
    }
  return dp;
}


//! Computes the dot product between two vectors in k-sparse representation. 
//! According to k-sparse representation, the indexes must be in ascending order, 
//! except for an optional terminal -1 that can be used to indicate that fewer than the length of the vector are used.
template<class T>
T ksparse_dot(const BVec<T>& u_values, const BVec<int>& u_indexes, const BVec<T>& v_values, const BVec<int>& v_indexes)
{
  // Faster handling of special case:
    if (u_indexes.length() == 1 && v_indexes.length() == 1)
    {
      if (u_indexes[0]==v_indexes[0])
        return u_values[0] * v_values[0];
      else
        return 0;
    }

  T dp = 0;
  int uk = 0, vk = 0;
  int ui = u_indexes[uk];
  int vi = v_indexes[vk];
  while( true )
    {
      if ( ui<vi ) // move forward in u indexes
        {
          uk++;
          if (uk >= u_indexes.length() ) // we've exhaisted u
            break;
          ui = u_indexes[uk];
          if (ui<0) // we've exhausted u
            break;
        }
      else if ( ui>vi ) // move forward in v indexes
        {
          vk++;
          if (vk >= v_indexes.length() ) // we've exhausted v
            break;
          vi = v_indexes[vk];
          if (vi<0)  // we've exhausted v
            break;
        }
      else // we have a match: ui==vi
        {
          dp += u_values[uk] * v_values[vk];
          uk++;
          vk++;
          if (uk >= u_indexes.length() || vk >= v_indexes.length() ) // we've exhausted u or v
            break;
          ui = u_indexes[uk];
          vi = v_indexes[vk];
          if (ui<0 || vi<0)  // we've exhausted u or v
            break;
        }
    }

  return dp;
}

//! Computes YTY = Y^T Y  with Y in column-K-sparse representation
//! Y is conceptually a n x m matrix in column K-sparse representation. 
//! So it is represented using two K x m matrices Y.values and Y.indexes.
//! The resulting Y^T Y is a dense m x m matrix.
//! Note that in column K-sparse representation, it is important that
//! each column of Y_indexes is sorted in ascending order,
//! (except for an optional terminal -1 indicating fewer than K positions are used).
template<class T>
void cksparse_square_mat(const CKSparseMat<T>& Y, const BMat<T>& YTY)
{
  assert(Y.values.nrows() == Y.indexes.nrows() && Y.values.ncols() == Y.indexes.ncols() );
  assert(YTY.nrows() == YTY.ncols());
  int m = Y.values.ncols();
  assert(YTY.nrows() == m);

  for (int i=0; i<m; i++)
  {
      YTY(i,i) = ksparse_squared_norm(Y.values.column(i), Y.indexes.column(i));
      for( int j=0; j<i; j++ )
        {
            T dp = ksparse_dot(Y.values.column(i), Y.indexes.column(i), Y.values.column(j), Y.indexes.column(j) );
            YTY(i,j) = dp;
            YTY(j,i) = dp;
        }
  }
}

/* y <- A x
   A is a dense m x n matrix in column major format. 
   x is a n dimensional K-sparse vector. Its K non-zero values and associated indexes are in x_values and x_indexes.
   The call will fill m-dimensional dense vector y
*/
template<class T>
void product_dense_mat_ksparse_vec(const BMat<T>& A, const KSparseVec<T>& x, const BVec<T>& y)
{
  y.clear();
  int K = x.indexes.length();
  for (int k=0; k<K; k++)
    {
      if (k<K-1 && x.indexes[k+1]>=0) // prefetch column of A indicated in next element of x.indexes
          prefetch_array(A.data() + A.stride() * x.indexes[k+1], A.nrows());

      int j = x.indexes[k];
      if (j<0) // we've exhausted x (-1 indicates optional early termination, when fewer than K indexes are to be used) 
        break;
      T val = x.values[k];
      blas_axpy(val, A.column(j), y);
    }
}

/* y <- (A,A,...,A) x
   A is a dense m x l matrix in column major format. 
   x is a n dimensional K-sparse vector. Its K non-zero values and associated indexes are in x_values and x_indexes.
   The call will fill m-dimensional dense vector y
*/
template<class T>
void product_repeated_dense_mat_ksparse_vec(const BMat<T>& A, const KSparseVec<T>& x, const BVec<T>& y)
{
  int ancols = A.ncols();
  y.clear();
  int K = x.indexes.length();
  for (int k=0; k<K; k++)
    {
      if (k<K-1 && x.indexes[k+1]>=0) // prefetch column of A indicated in next element of x.indexes
          prefetch_array(A.data() + A.stride() * (x.indexes[k+1] % ancols), A.nrows());

      int j = x.indexes[k] % ancols;
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
*/
template<class T>
void product_dense_mat_cksparse_mat(const BMat<T>& A,
                                    const CKSparseMat<T>& X,
                                    const BMat<T>& Y)
{
  assert(Y.nrows()==A.nrows() && Y.ncols()==X.ncols());
  assert(A.ncols() == X.nrows());

  int l = X.ncols();

  for (int j=0; j<l; j++)
    {
      if (j<l-1) // prefetch column of A indicated in first element of next column of X.indexes
        {
          int nextcol =  X.indexes(0,j+1);
          if (nextcol>=0)
              prefetch_array(A.data() + A.stride() * nextcol, A.nrows());
        }
      product_dense_mat_ksparse_vec(A, X.column(j), Y.column(j));
    }
}

/* Y <- (A,A,...,A) X
   A is a dense (m,n') matrix in column major format. 
   X is a (n,l) column-K-sparse matrix. 
   For each of its l columns, its K non-zero values and associated indexes are in the corresponding column of X_values and X_indexes (booth are (K,l) matrices ).
   The call will fill (m,l) dimensional dense matrix Y
*/
template<class T>
void product_repeated_dense_mat_cksparse_mat(const BMat<T>& A,
                                    const CKSparseMat<T>& X,
                                    const BMat<T>& Y)
{
  assert(Y.nrows()==A.nrows() && Y.ncols()==X.ncols());
  assert(A.ncols() == X.nrows());

  int l = X.ncols();

  for (int j=0; j<l; j++)
    {
      if (j<l-1) // prefetch column of A indicated in first element of next column of X.indexes
        {
          int nextcol =  X.indexes(0,j+1);
          if (nextcol>=0)
              prefetch_array(A.data() + A.stride() * nextcol, A.nrows());
        }
      product_repeated_dense_mat_ksparse_vec(A, X.column(j), Y.column(j));
    }
}


// A <- A + u v^T
// A is a dense (m,n) matrix in column major representation
// u is a dense m-dimensional vector
// v is a sparse n-dimensional vector in K-sparse representation
template<class T>
void rank_one_update_dense_vec_ksparse_vec(const BVec<T>& u, const KSparseVec<T>& v, const BMat<T>& A)
{
  int K = v.K();
  for(int k=0; k<K; k++)
    {
      if (k<K-1 && v.indexes[k+1]>=0)
          prefetch_array(A.data() + A.stride() * v.indexes[k+1], A.nrows());

      int j = v.indexes[k];
      if (j<0) // we've exhausted v (-1 indicates optional early termination, when fewer than K indexes are to be used) 
            break;        
      blas_axpy(v.values[k], u, A.column(j) );
    }
}

// A <- A + U V^T  Does a rank l update to matrix A.
// A is a dense (m,n) matrix in column major representation
// U is a dense (m,l) matrix in column major representation
// V is a sparse (n,l) matrix in column-K-sparse representation
template<class T>
void rank_update_dense_mat_cksparse_mat(const BMat<T>& U,
                                        const CKSparseMat<T>& V, 
                                        const BMat<T>& A)
{
  assert( A.nrows() == U.nrows() && A.ncols()==V.nrows() && U.ncols() == V.ncols() );
 
  // loop over rank-one updates
  int l = U.ncols();

  for(int j=0; j<l; j++)
    {
      if (j<l-1) // prefetch column of A indicated in first element of next column of V.indexes
        {
          int nextcol = V.indexes(0,j+1);
          if (nextcol>=0)
              prefetch_array(A.data() + A.stride()*nextcol, A.nrows());
        }
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
