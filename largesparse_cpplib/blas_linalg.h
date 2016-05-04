// -*- C++ -*-

// blas_linalg.h
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
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TOR (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


#ifndef blas_linalg_INC
#define blas_linalg_INC

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include "plerror.h"
#include "pl_math.h"

#include <algorithm> 
#include "CpuStorage.h"
#include "blas_proto_overload.h"
#include "lapack_proto_overload.h"


namespace PLearn {

using std::max;
using std::min;

/*
// max and min for different types
template <class T> T max(const T& v1, const T& v2) 
{ return (v1>v2)?v1 :v2; }

// max and min for different types
template <class T> T min(const T& v1, const T& v2) 
{ return (v1<v2)?v1 :v2; }
*/


// low-level function, might need specialization for only low-level types (?) and generic version for all others.
template<class T>
inline void memcpy_contiguous_array(T* dest, T* src, int n)
{ memcpy(dest, src, n*sizeof(T)); }

//! Sets ptr to 0 (generic version, followed by specialized fast versions)
template<class T>
inline void memclear(T* ptr, int n)
{ 
  for(int i=0; i<n; i++)
    ptr[i] = 0;
}

inline void memclear(int* ptr, int n)
{ memset(ptr, 0, n*sizeof(int)); }

inline void memclear(float* ptr, int n)
{ memset(ptr, 0, n*sizeof(float)); }

inline void memclear(double* ptr, int n)
{ memset(ptr, 0, n*sizeof(double)); }


// absolute vale for different types
inline int myabs(int x) { return abs(x); }
inline float myabs(float x) { return fabsf(x); }
inline double myabs(double x) { return fabs(x); }
inline long double myabs(long double x) { return fabsl(x); }


// low level sorting for contiguous arrays of int, float and double

int compare_int(const void* a, const void* b)
{
  int* fa = (int*) a;
  int* fb = (int*) b;
  return *fa - *fb;
}

int compare_float(const void* a, const void* b)
{
  float* fa = (float*) a;
  float* fb = (float*) b;
  return *fa - *fb;
}

int compare_double(const void* a, const void* b)
{
  double* fa = (double*) a;
  double* fb = (double*) b;
  return *fa - *fb;
}

inline void sort(int* x, int n)
{ qsort( x, n, sizeof(int), compare_int); }

inline void sort(float* x, int n)
{ qsort( x, n, sizeof(float), compare_float); }

inline void sort(double* x, int n)
{ qsort( x, n, sizeof(double), compare_double); }


// Default fixed formatted printing of basic types (will be called when printing matrices and vectors )
inline void print(int x) { printf("%10d",x); }
inline void print(float x) { printf("%15.6e",x); }
inline void print(double x) { printf("%15.6e",x); }

// Note: semantic of const for BVec and BMat classes: 
// A vector or matrix is considered to remain const as long as its *structure* does not change
// The data content of a const& BVec or BMat can be changed.
// This allows ex to pass A.column(3) directly as the argument of a function taking in a const BVec& 
// without the compiler complaining, and avoids extra complications.
// So in summary a const BVec or BMat can have its data changed, but not its structure (dimensions, ...)

// predeclarations
template<class T> class BMat;

// *************************
// *   BVec class
// *************************
// This BVec class is meant to be but a thin wrapper around a blas vector residing in host memory

template<class T> 
class BVec {

    friend class BMat<T>;

protected:
    T* data_;     // pointer to first element 
    int length_;  // number of elements
    int step_;    // step to next element of vector 
    PP< CpuStorage<T> > storage_;  // smartpointer to allocated memory
    
public:

    typedef T elem_t;  // element type
    
    inline T* data() const
    { return data_; }

    inline const T* const_data() const
    { return data_; }

    inline int length() const
    { return length_; }
    
    inline int step() const
    { return step_; }

    //! Default constructor
    inline BVec()
        :data_(0), length_(0), step_(0), storage_(0)
    {}


  //! constructor from existing already allocated memory chunk. Not managed by storage.
    inline BVec(T* data, int length, int step, CpuStorage<T>* storage=0)
      :data_(data), length_(length), step_(step), storage_(storage)
  {}

  //! copy constructor. Copies the structure, not the data.
  inline BVec(const BVec<T>& x)
      :data_(x.data_), length_(x.length_), step_(x.step_), storage_(x.storage_)
  {}

    //! Returns true if step==1 (the elements of the vector are contiguous in memory)
    inline bool is_contiguous() const 
    { return step_==1; }

    inline bool isNull() const 
    { return data_==0; }

    inline bool isNotNull() const
    { return data_!=0; }

    inline bool isEmpty() const
    { return length_ == 0; }

    inline bool isNotEmpty() const
    { return length_ != 0; }


  //! constructor that allocates memory
    inline explicit BVec(int length, MemoryType memtype_=CUDA_HOST)
    :length_(length), step_(1)
  {
      storage_ = new CpuStorage<T> (length_, memtype_);  // Will be automatically deleted when its refcount becomes zero
      assert( storage_->refcount() == 1);
      data_ = storage_->dataptr_;

      // data_ = new T[length];
      // storage = 0;
  }

  //! operator= Copies the structure, not the data 
  // (use operator << if you want to copy the data between two vectors)
  inline const BVec<T>& operator=(const BVec<T>& x)
  {
    data_ = x.data_;
    length_ = x.length_;
    step_ = x.step_;
    storage_ = x.storage_;
    return *this;
  }
  
  //! y << x   (copies elements of x to this vector y) vectors must not overlap
  inline void operator<< (const BVec<T>& x) const
  {
    assert(this->length_==x.length_);
    if ( this->is_contiguous() && x.is_contiguous() )
      memcpy_contiguous_array(this->data_, x.data_, x.length_);
    else
      {
        T* xptr = x.data_;
        T* yptr = this->data_;
        for(int i=0; i<x.length_; i++, xptr+=x.step_, yptr+=this->step_)
          *yptr = *xptr;
      }
  }

  //! x >> y   (copies elements of this vector x to y) vectors must not overlap
  inline void operator>> (const BVec<T>& y) const 
    { y.operator<<(*this); }

    //! Resizes this vector, reallocating memory if necessary
    //! If storage has enough memory already, this is just changing the length_ field.
    //! Otherwise new contiguous memory storage will be allocated and the old data values 
    //! will be copied over to that new storage (provided copy_data=true which is the default). 
    //! If you do not need to preserve the old data, pass false to copy_data 
    //! and no time will be wasted copying it over
    void resize(int new_length, bool copy_data=true)
    {
        assert(new_length>=0);
        
        if (new_length != length_)
        {
            if (data_==0)  // we've never allocated this vec yet
            {
                (*this) = BVec(new_length);
            }
            else if( new_length<length_ )
            {
                length_ = new_length;
            }
            else if (storage_!=0)
            {
                if (data_+new_length*step_ <= storage_->dataptr_+storage_->n_)
                {
                    length_ = new_length;
                }
                else // must reallocate and copy
                {
                    if (storage_->refcount()>1) 
                        PLERROR("Cannot safely reallocate storage elsewhere: it has already more than one reference pointing to it");
                    BVec<T> newv(new_length, storage_->memtype_);
                    if (copy_data)
                        newv.subVec(0,length_) << (*this);
                    (*this) = newv;
                }
            }
            else
            { PLERROR("Resizing vector to larger size not supported if there's no storage"); }
        }
    }

  //! fills the vector with the specified element
  void fill(T val) const
  {
    T* ptr = data_;
    for(int i=0; i<length_; i++, ptr += step_)
      *ptr = val;
  }

  //! fills the vector with zeros
  inline void clear() const
  { 
      if (is_contiguous())
        memclear(data_, length_); 
    else
        fill(0);
  }


  //! access element at position i (0-based indexing)
  inline T& operator[](int i) const
  {
    assert(i>=0 && i<length_);
    return data_[i*step_];
  }

  //! access element at position i (0-based indexing)
  inline T& operator()(int i) const
  { return operator[](i); }
  
  //! returns a sub-vector
  inline BVec<T> subVec(int startpos, int subvec_length) const
  {
    assert(subvec_length>=0 && startpos+subvec_length <=length_);
    BVec<T> subv(data_+startpos*step_, subvec_length, step_, storage_);
    return subv;
  }

    //! Returns a vector (allocated in main host memory) filled with 0, of the specified length
    //! Do NEVER modify the elements of this vector, as other calls will rely on it being filled with zeros
    static BVec<T> zeros(int d)
    {
        static BVec<T> host_zeros;
        if ( d>host_zeros.length() )
        {
            int n = max(d*2, 1024);
            host_zeros = BVec<T>(n);
            host_zeros.clear();
        }
        return host_zeros.subVec(0,d);                        
    }

    //! Returns a vector (allocated in main host memory) filled with 1 of the specified length
    //! Do NEVER modify the elements of this vector, as other calls will rely on it being filled with ones
    static BVec<T> ones(int d)
    {
        static BVec<T> host_ones;
        if ( d>host_ones.length() )
        {
            int n = max(d*2, 1024);
            host_ones = BVec<T>(n);
            host_ones.fill( T(1) );
        }
        return host_ones.subVec(0,d);                        
    }

    //! This will return a view of the kth tmpvec, which will have length d
    //! Call this to avoid having to allocate your own temporary vecs for temporry computations
    //! BEWARE: Be EXTREMELY careful not to call a function which makes use of the same kth tmpvec and changes it without you suspecting it
    static BVec<T> tmpvec(int d, int k=0)
    {
        static BVec<T> host_tmpvecs[10];

        assert(k<10);
        BVec<T>& host_tmpvec = host_tmpvecs[k];
        if ( d>host_tmpvec.length() )
        {
            int n = max(d*2, 1024);
            host_tmpvec = BVec<T>(n);
        }
        return host_tmpvec.subVec(0,d);                        
    }



};

// *************************
// *   BMat class
// *************************
// This BMat class is meant to be but a thin wrapper around a column-major blas matrix residing in host memory

template<class T>
class BMat {

protected:
    T* data_; /* pointer to first element of matrix in column-major storage */
    int nrows_;
    int ncols_;
    int stride_; /* step to next column (setp to next row is always 1 in blas column major matrices) */
    PP< CpuStorage<T> > storage_;  // smartpointer to allocated memory

public:

    typedef T elem_t;  // type of elements
    typedef BVec<T> vec_t; // associated vector type (what's returned e.g. by row(i) and column(j) ) 
    
    inline T* data() const { return data_; }
    inline const T* const_data() const { return data_; }
    inline int nrows() const { return nrows_; }
    inline int ncols() const { return ncols_; }
    inline int stride() const { return stride_; }
    
  //! returns true if the matrix elements are contiguous in memory
  //! This will be the case if nrows==1 (since in row-major format, all the elems of a row are contiguous)
  //! or when stride==nrows
  inline bool is_contiguous() const 
  { return stride_==nrows_ || ncols_==1; }
  
    inline bool isNull() const 
    { return data_==0; }

    inline bool isNotNull() const
    { return data_!=0; }

    inline bool isEmpty() const
    { return nrows_ == 0 || ncols_ == 0; }

    inline bool isNotEmpty() const
    { return nrows_ != 0 && ncols_ != 0; }


  //! Default constructor
  inline BMat()
      :data_(0), nrows_(0), ncols_(0), stride_(0), storage_(0)
  {}
  
  //! constructor from existing already allocated memory chunk (if you do not want this to be automatically memory managed, pass 0 for storage)
    inline BMat(T* data, int nrows, int ncols, int stride, CpuStorage<T>* storage=0)
      :data_(data), nrows_(nrows), ncols_(ncols), stride_(stride), storage_(storage)
  {
      assert(data!=0 || (nrows==0 && ncols==0) );
  }

  //! copy constructor. Copies the structure, not the data. Results in another view on the same data.
  inline BMat(const BMat<T>& m)
      :data_(m.data_), nrows_(m.nrows_), ncols_(m.ncols_), stride_(m.stride_),
       storage_(m.storage_)
  {}

    //! View of a contiguous vector as a single column matrix or single row matrix
    inline BMat(const BVec<T>& x, bool as_row=false)
      :data_(x.data()), nrows_(x.length()), ncols_(1), stride_(x.length()), storage_(x.storage_)
  {
      if (!as_row)  // view it a sa single column matrix, must be contiguous
          assert( x.is_contiguous() );
      else // view it a sa single row matrix
      {
          nrows_ = 1;
          ncols_ = x.length_;
          stride_ = x.step_;
      }      
  }

  //! Constructs a matrix view of a contiguous vector
  inline BMat(const BVec<T>& x, int nrows, int ncols)
      :data_(x.data()), nrows_(nrows), ncols_(ncols), stride_(nrows), storage_(x.storage_)
  {
    assert( x.is_contiguous() );
    assert( nrows*ncols==x.length() );
  }

  //! Returns a flat vector view of a contiguous matrix
  inline BVec<T> flat() const
  {
    assert( is_contiguous() );
    return BVec<T>(data_, nrows_*ncols_, 1, storage_);
  }

    //! constructor that allocates memory
    inline BMat(int nrows, int ncols,  MemoryType memtype=CUDA_HOST)
        :nrows_(nrows), ncols_(ncols), stride_(nrows_)
    {
        assert(nrows>=1 && ncols>=1);
        
        storage_ = new CpuStorage<T> (nrows*ncols, memtype);  // Will be automatically deleted when its refcount becomes zero
        assert( storage_->refcount() == 1);
        data_ = storage_->dataptr_;
        
        //data_ = new T[nrows*ncols];
        //storage_ = 0;
    }


  //! operator= Copies the structure, not the data 
  // (use operator << if you want to copy the data between two vectors)
  inline const BMat<T>& operator=(const BMat<T>& m)
  {
    data_ = m.data_;
    nrows_ = m.nrows_;
    ncols_ = m.ncols_;
    stride_ = m.stride_;
    storage_ = m.storage_;
    return *this;
  }

    //! C <- A (copies elements) matrices must not overlap
    inline void operator<< (const BMat<T>& A) const
    {
        assert( A.nrows_==nrows_ && A.ncols_==ncols_ );
        // Detect contiguous memory, if so we can copy in one shot
        if ( A.is_contiguous() && this->is_contiguous() )
            memcpy_contiguous_array(this->data_, A.data_, A.nrows_*A.ncols_);
        else // Non contiguous, need to copy column by column
        {
            T* Aj = A.data_;
            T* Cj = this->data_;
            for (int j=0; j<A.ncols_; j++, Aj+=A.stride_, Cj+=this->stride_)
                memcpy_contiguous_array(Cj, Aj, A.ncols_);
        }
    }

  //! A >> C  (copies elements of this matrix A to matrix C) matrices must not overlap
  inline void operator>> (const BMat<T>& C) const
    { C.operator<<(*this); }

    //! Resizes this matrix, reallocating memory if necessary
    //! If storage has enough memory already no new memory is allocated
    //! Otherwise new contiguous memory storage will be allocated and the old data values 
    //! will be copied over to that new storage (provided copy_data=true which is the default). 
    //! If you do not need to preserve the old data, pass false to copy_data 
    //! and no time will be wasted copying data over
    void resize(int new_nrows, int new_ncols, bool copy_data=true)
    {
        assert(new_nrows>=0);
        assert(new_ncols>=0);

        if (data_==0)  // we've neve allocated this Mat yet
        {
            (*this) = BMat<T>(new_nrows, new_ncols);
        }
        else if (new_nrows<=stride_ && new_ncols<=ncols_) 
        {
            nrows_ = new_nrows;
            ncols_ = new_ncols;
        }
        else if (storage_!=0) // verify if there's enough room in storage to grow
        {
            if (new_nrows<=stride_ && data_+new_ncols*stride_ <= storage_->dataptr_+storage_->n_)
            {
                nrows_ = new_nrows;
                ncols_ = new_ncols;
            }
            else // must reallocate and copy
            {
                if (storage_->refcount()>1) 
                    PLERROR("Cannot safely reallocate storage elsewhere: it has already more than one reference pointing to it");                
                BMat<T> newm(new_nrows, new_ncols, storage_->memtype_);
                if (copy_data)
                    newm.subMat(0,0,nrows_,ncols_) << (*this);
                (*this) = newm;
            }
        }
        else
        {
            PLERROR("Resizing matrix to larger size not supported if there's no storage");
        }
    }



    //! fills the matrix with the specified element
    inline void fill(T val) const
    {
        if (is_contiguous())
        {
            for(int i=0; i<nrows_*ncols_; i++)
                data_[i]= val;
        }
        else
        {
            T* col_j = data_;
            for(int j=0; j<ncols_; j++, col_j+=stride_)
            {
                for(int i=0; i<nrows_; i++)
                    col_j[i] = val;
            }
        }
    }

    //! fills the matrix with zeros
    inline void clear() const
    { 
        if (is_contiguous())
            memclear( data_, nrows_*ncols_ );
        else
        {
            T* col_j = data_;
            for(int j=0; j<ncols_; j++, col_j+=stride_)
                memclear( col_j, nrows_ );
        }
    }
    
    //! access element at position i (0-based index)
    inline T& operator() (int i, int j) const
    {
      assert(i>=0 && i<nrows_ && j>=0 && j<ncols_);
      return data_[i+stride_*j];
  }
    

    //! read-only access to element at position i (0-based index)
    inline T get(int i, int j) const
    {
        assert(i>=0 && i<nrows_ && j>=0 && j<ncols_);
        return data_[i+stride_*j];
    }

    inline void set(int i, int j, const T& val) const
    {
      assert(i>=0 && i<nrows_ && j>=0 && j<ncols_);
      data_[i+stride_*j] = val;
    }

    //! returns a vector that is a row of this matrix
      // index argument is 0 based (i.e. between 0 and nrows-1 inclusive)
      inline BVec<T> row(int i) const
      {
        assert(i>=0 && i<nrows_);
        return BVec<T>(data_+i, ncols_, stride_, storage_);
      }
  
    //! returns a vector that is a column of this matrix
    // index argument is 0 based (i.e. between 0 and ncols-1 inclusive)
    inline BVec<T> column(int j) const
    {
        assert(j>=0 && j<ncols_);
        return BVec<T>(data_+j*stride_, nrows_, 1, storage_);
    }
  
    //! Returns a BVec view of the main diagonal of this matrix 
    inline BVec<T> diagonal() const
    {
        BVec<T> di( data_, min(nrows_, ncols_), stride_+1, storage_ );
        return di;
    }

    //! Returns a BVec view of the main diagonal of this matrix (alias for diagonal() )
    inline BVec<T> diag() const
    { return diagonal(); }

    //! returns a sub-matrix
    // index arguments i,j are 0 based 
    inline BMat<T> subMat(int i, int j, int nrows, int ncols) const
    {
      assert(i>=0 && j>=0 && nrows>=1 && ncols>=1 && i+nrows<=nrows_ && j+ncols<=ncols_);
      BMat<T> subm(data_+i+j*stride_, nrows, ncols, stride_, storage_);
      return subm;
    }

    //! returns a sub-matrix corresponding to a range of full columns
    // index arguments j is 0 based 
    inline BMat<T> subMatColumns(int j, int ncols) const
    {
      assert(j>=0 && ncols>=1 && j+ncols<=ncols_);
      BMat<T> subm(data_+j*stride_, nrows_, ncols, stride_, storage_);
      return subm;
    }

    //! returns a sub-matrix corresponding to a range of full rows
    // index arguments i is 0 based 
    inline BMat<T> subMatRows(int i, int nrows) const
    {
      assert(i>=0 && nrows>=1 && i+nrows<=nrows_);
      BMat<T> subm(data_+i, nrows, ncols_, stride_, storage_);
      return subm;
    }
  
  };


  /* ******************************************************* */
  /*   Define generic functions for BVec<T> and BMat<T>        */
  /* ******************************************************* */
 
  // sorting
  template<class T>
  void sort(const BVec<T>& x)
  { 
    if (!x.is_contiguous())
      {
        printf("Sorting of vector eements not yet implemented in non-contiguous case");
        exit(1);        
      }
    sort(x.data(), x.length());
  }

  // -------------------------
  // Basic I/O
  
  template<class T> 
  void print(const BVec<T>& x)
  {
      printf("BVec (n=%d, step=%d): [  ", x.length(), x.step());
      for (int i=0; i<x.length(); i++)
      {
        print(x[i]);
        printf(" ");
      }
    printf("   ]  \n");
  }

  template<class T> 
  void print(const BMat<T>& A)
  {
      printf("BMat %d x %d  (stride=%d)\n", A.nrows(), A.ncols(), A.stride());
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




  // -----------------------
  // Random filling


  //! Fills vector with integers uniformly sampled between low and high inclusive 
  template<class T>
  void fill_random_uniform_discrete(const BVec<T>& x, int low, int high)
  {
    int range = high-low+1;
    for (int i=0; i<x.length(); i++)
      {
        int ir = rand(); 
        x[i] = T(low + ir%range);
      }
  }

  //! Fills vector with integers uniformly sampled between low and high inclusive, keeps resampling if we already sampled the value
  template<class T>
  void fill_random_uniform_discrete_noduplicate(const BVec<T>& x, int low, int high)
  {
    int range = high-low+1;
    for (int i=0; i<x.length(); i++)
      {
        int val = 0;
        bool unique;
        do
          {
            int ir = rand(); 
            val = low + ir%range;
            unique = true;
            int k;
            for (k=0; k<i; k++)
              if (x[k] == T(val) )
                {
                  unique = false;
                  break;
                }
          } while (!unique);

        x[i] = T(val);
      }
  }

  //! Fills matrix with integers uniformly sampled between low and high inclusive 
  template<class T>
  void fill_random_uniform_discrete(const BMat<T>& m, int low, int high)
  {
    if (m.is_contiguous())
      fill_random_uniform_discrete(m.flat(), low, high);
    else
      {
          for(int j=0; j<m.ncols(); j++)
          fill_random_uniform_discrete(m.column(j), low, high);
      }
  }

  //! Fills vector with real numbers (converted to T) uniformly sampled between low and high 
  template<class T>
  void fill_random_uniform_continuous(const BVec<T>& x, double low, double high)
  {
    double range = high-low;
    for (int i=0; i<x.length(); i++)
      {
        int ir = rand(); 
        x[i] = T(low + range*(((double)ir)/RAND_MAX));
      }
  }

  //! Fills matrix A with real numbers (converted to T) uniformly sampled between low and high 
  template<class T>
  void fill_random_uniform_continuous(const BMat<T>& A, double low, double high)
  {
    if (A.is_contiguous())
      fill_random_uniform_continuous(A.flat(), low, high);
    else
      {
          for(int j=0; j<A.ncols(); j++)
          fill_random_uniform_continuous(A.column(j), low, high);
      }
  }


  // ------------------------------------
  // Blas Linear algebra operations 


  //! returns x' y  
template<class T>
inline T blas_dot(const BVec<T>& x, const BVec<T>& y)
{ 
    assert(x.length()==y.length()); 
    return blas_dot( x.length(), x.data(), x.step(), y.data(), y.step() ); 
}

// void blas_dot(n, const BVec<T>& x, const BVec<T>& y, resultptr) sdot_sub_( &x.length, x.data, &x.step, y.data, &y.step, resultptr );  

//! x = alpha x   
template<class T>
inline void blas_scal(T alpha, const BVec<T>& x)
{
    blas_scal(x.length(), alpha, x.data(), x.step() ); 
}

template<class T, class Talpha>
inline void blas_scal(Talpha alpha, const BVec<T>& x)
{ blas_scal( T(alpha), x); }

template<class T, class Talpha>
inline void operator*=(const BVec<T>& x, Talpha alpha)
{ blas_scal( T(alpha), x); }

//! x += a
template<class T, class Talpha>
inline void operator+=(const BVec<T>& x, Talpha alpha)
  {
      T* x_data = x.data();
      int n = x.length();
      int step = x.step();
      while(n--)
      {
          *x += alpha;
          x += step;
      }
  } 

//! x -= a
template<class T, class Talpha>
inline void operator-=(const BVec<T>& x, Talpha alpha)
  {
      operator+=(x, -alpha);
  } 


template<class T>
inline void operator*=(const BVec<T>& x, const BVec<T>& y)
{ 
    T* x_data = x.data();
    int x_step = x.step();
    const T* y_data = y.const_data();
    int y_step = y.step();
    int n = x.length();
    assert(y.length()==n);
    while(n--)
    {
        *x_data *= *y_data;
        x_data += x_step;
        y_data += y_step;
    }
}

template<class T, class Talpha>
inline void operator/=(const BVec<T>& x, Talpha alpha)
{ blas_scal( 1/T(alpha), x); }


//! A = alpha A
template<class T>
inline void blas_scal(T alpha, const BMat<T>& A)
{ 
    if (A.is_contiguous())          
        blas_scal(alpha, A.flat() ); 
    else
    {
        for (int j=0; j<A.ncols(); j++)
            blas_scal(alpha, A.column(j));
    }
}

template<class T, class Talpha>
inline void blas_scal(Talpha alpha, const BMat<T>& A)
{ blas_scal( T(alpha), A); }

template<class T, class Talpha>
inline void operator*=(const BMat<T>& A, Talpha alpha)
{ blas_scal( T(alpha), A); }

template<class T, class Talpha>
inline void operator/=(const BMat<T>& A, Talpha alpha)
{ blas_scal( 1/T(alpha), A); }

  //! y = y + alpha x 
  template<class T>
  inline void blas_axpy(T alpha, const BVec<T>& x, const BVec<T>& y)
  { 
      assert(x.length()==y.length()); 
      blas_axpy(x.length(), alpha, x.data(), x.step(), y.data(), y.step()); 
  }

  template<class T, class Talpha>
  inline void blas_axpy(Talpha alpha, const BVec<T>& x, const BVec<T>& y)
  { blas_axpy( T(alpha), x, y); }

//! x += a
  template<class T>
  inline void operator+=(const BVec<T>& x, const T& a)
  {
      T* x_data = x.data();
      int n = x.length();
      int step = x.step();
      while(n--)
      {
          *x_data += a;
          x_data += step;
      }
  } 

//! x -= y
  template<class T>
  inline void operator-=(const BVec<T>& x, const BVec<T>& y)
  { blas_axpy( T(-1), y, x); } 

//! x += y
  template<class T>
  inline void operator+=(const BVec<T>& x, const BVec<T>& y)
  { blas_axpy( T(1), y, x); } 

  //! Y = Y + alpha X 
  template<class T>
  inline void blas_axpy(T alpha, const BMat<T>& X, const BMat<T>& Y)
  { 
      assert(X.nrows()==Y.nrows() && X.ncols()==Y.ncols()); 
    if ( X.is_contiguous() && Y.is_contiguous() )
      blas_axpy( alpha, X.flat(), Y.flat() );
    else
      {
          for( int j=0; j<X.ncols(); j++)
          blas_axpy( alpha, X.column(j), Y.column(j) );
      }
  }

  template<class T, class Talpha>
  inline void blas_axpy(Talpha alpha, const BMat<T>& X, const BMat<T>& Y)
  { blas_axpy( T(alpha), X, Y); }

//! X -= Y
  template<class T>
  inline void operator-=(const BMat<T>& X, const BMat<T>& Y)
  { blas_axpy( T(-1), Y, X); } 

//! X += Y
  template<class T>
  inline void operator+=(const BMat<T>& X, const BMat<T>& Y)
  { blas_axpy( T(+1), Y, X); } 

//! X += y (y considered a column vector)
  template<class T>
  inline void addColumnVector(const BMat<T>& X, const BVec<T>& y)
  {
      assert(y.length()==X.nrows());
      int ncols = X.ncols();
      for (int j=0; j<ncols; j++)
          X.column(j) += y;
  }

//! X += y^T (y^T considered a row vector)
  template<class T>
  inline void addRowVector(const BMat<T>& X, const BVec<T>& y)
  {
      int ncols = X.ncols();
      assert(y.length()==ncols);
      for (int j=0; j<ncols; j++)
          X.column(j) += y[j];
  }

//! X += y (same as addColumnVector)
  template<class T>
  inline void operator+=(const BMat<T>& X, const BVec<T>& y)
  {
      addColumnVector(X,y);
  }



  //! A = A + alpha x y'  
  template<class T>
  inline void blas_ger(T alpha, const BVec<T>& x, const BVec<T>& y, const BMat<T>& A)
  { 
      assert(A.nrows()==x.length() && A.ncols()==y.length());
      blas_ger(x.length(), y.length(), alpha, x.data(), x.step(), y.data(), y.step(), A.data(), A.stride());
  }

  template<class T, class Talpha>
  inline void blas_ger(Talpha alpha, const BVec<T>& x, const BVec<T>& y, const BMat<T>& A)
  { blas_ger(T(alpha), x, y, A); }


  //! A = A + alpha x x'    
  //! A is supposed symmetric and we use only its lower part (if uplo='L') or upper part (if uplo='U') 
  template<class T>
  inline void blas_syr(char uplo, T alpha, const BVec<T>& x, const BMat<T>& A)
  { 
    assert( uplo=='L' || uplo=='U' );
    assert(A.nrows()==x.length() && A.ncols()==x.length()); 
    blas_syr(uplo, x.length(), alpha, x.data(), x.step(), A.data(), A.stride()); 
  }

  template<class T, class Talpha>
  inline void blas_syr(char uplo, Talpha alpha, const BVec<T>& x, const BMat<T>& A)
  { blas_syr(uplo, T(alpha), x, A); }

  //! A = A + alpha x y' + alpha y x'   
  //! A is supposed symmetric, considering only its lower part (if uplo='L') or upper part (if uplo='U') 
  template<class T>
  inline void blas_syr2(char uplo, T alpha, const BVec<T>& x, const BVec<T>& y, const BMat<T>& A)
  { 
    assert( uplo=='L' || uplo=='U' );
    assert(x.length()==y.length() && A.nrows()==x.length() && A.ncols()==A.nrows()); 
    blas_syr2(uplo, x.length(), alpha, x.data(), x.step(), y.data(), y.step(), A.data(), A.stride()); 
  }

  template<class T, class Talpha>
  inline void blas_syr2(char uplo, Talpha alpha, const BVec<T>& x, const BVec<T>& y, const BMat<T>& A)
  { blas_syr2(uplo, T(alpha), x, y, A); }

  //! y = alpha A x + beta y   (if trans=='N')
  //! y = alpha A^T x + beta y   (if trans=='T')
  template<class T>
  inline void blas_gemv(char trans, T alpha, const BMat<T>& A, const BVec<T>& x, T beta, const BVec<T>& y)
  { 
      assert( (trans=='N' && A.nrows()==y.length() && A.ncols()==x.length())  || 
              (trans=='T' && A.nrows()==x.length() && A.ncols()==y.length()) );
      blas_gemv(trans, A.nrows(), A.ncols(), alpha, A.data(), A.stride(), x.data(), x.step(), beta, y.data(), y.step() ); 
  }

  template<class T, class Talpha, class Tbeta>
  inline void blas_gemv(char trans, Talpha alpha, const BMat<T>& A, const BVec<T>& x, Tbeta beta, const BVec<T>& y)
  { blas_gemv(trans, T(alpha), A, x, T(beta), y); }

  //! y = alpha A x + beta y  
  //! A is a symmetric matrix, and we use only its lower part (if uplo='L') or upper part (if uplo='U') 
  template<class T>
  inline void blas_symv(char uplo, T alpha, const BMat<T>& A, const BVec<T>& x, T beta, const BVec<T>& y)
  { 
    assert( uplo=='L' || uplo=='U' );
    assert(A.nrows()==A.ncols() && A.ncols()==x.length() && A.nrows()==y.length()); 
    blas_symv(uplo, y.length(), alpha, A.data(), A.stride(), x.data(), x.step(), beta, y.data(), y.step() ); 
  }

  template<class T, class Talpha, class Tbeta>
  inline void blas_symv(char uplo, Talpha alpha, const BMat<T>& A, const BVec<T>& x, Tbeta beta, const BVec<T>& y)
  { blas_symv(uplo, T(alpha), A, x, T(beta), y); }

  //! C = alpha A B + beta C 
  template<class T>
  inline void blas_gemm_NN(T alpha, const BMat<T>& A, const BMat<T>& B, T beta, const BMat<T>& C)
  {  
      assert(A.nrows()==C.nrows() && B.ncols()==C.ncols() && A.ncols()==B.nrows());
    blas_gemm('N', 'N', C.nrows(), C.ncols(), A.ncols(),            
              alpha, A.data(), A.stride(), B.data(), B.stride(), beta, C.data(), C.stride()); 
  }

  template<class T, class Talpha, class Tbeta>
  inline void blas_gemm_NN(Talpha alpha, const BMat<T>& A, const BMat<T>& B, Tbeta beta, const BMat<T>& C)
  { blas_gemm_NN(T(alpha), A, B, T(beta), C); }

  //! C = alpha A' B + beta C 
  template<class T>
  inline void blas_gemm_TN(T alpha, const BMat<T>& A, const BMat<T>& B, T beta, const BMat<T>& C)
  {  
      assert(A.ncols()==C.nrows() && B.ncols()==C.ncols() && A.nrows()==B.nrows()); 
      blas_gemm('T', 'N', C.nrows(), C.ncols(), A.nrows(), 
              alpha, A.data(), A.stride(), B.data(), B.stride(), beta, C.data(), C.stride()); 
  }

  template<class T, class Talpha, class Tbeta>
  inline void blas_gemm_TN(Talpha alpha, const BMat<T>& A, const BMat<T>& B, Tbeta beta, const BMat<T>& C)
  { blas_gemm_TN(T(alpha), A, B, T(beta), C); }

//! C = alpha A B' + beta C 
template<class T>
inline void blas_gemm_NT(T alpha, const BMat<T>& A, const BMat<T>& B, T beta, const BMat<T>& C)
{  
    assert(A.nrows()==C.nrows() && B.nrows()==C.ncols() && A.ncols()==B.ncols());
    blas_gemm('N', 'T', C.nrows(), C.ncols(), A.ncols(), 
              alpha, A.data(), A.stride(), B.data(), B.stride(), beta, C.data(), C.stride()); 
}

  template<class T, class Talpha, class Tbeta>
  inline void blas_gemm_NT(Talpha alpha, const BMat<T>& A, const BMat<T>& B, Tbeta beta, const BMat<T>& C)
  { blas_gemm_NT(T(alpha), A, B, T(beta), C); }

  //!  C = alpha A' B' + beta C 
  template<class T>
  inline void blas_gemm_TT(T alpha, const BMat<T>& A, const BMat<T>& B, T beta, const BMat<T>& C)
  {  
    assert(A.ncols()==C.nrows() && B.nrows()==C.ncols() && A.nrows()==B.ncols());
    blas_gemm('T', 'T', C.nrows(), C.ncols(), A.nrows(), 
              alpha, A.data(), A.stride(), B.data(), B.stride(), beta, C.data(), C.stride()); 
  }


  template<class T, class Talpha, class Tbeta>
  inline void blas_gemm_TT(Talpha alpha, const BMat<T>& A, const BMat<T>& B, Tbeta beta, const BMat<T>& C)
  { blas_gemm_TT(T(alpha), A, B, T(beta), C); }

  //! C = alpha A B + beta C 
  //! A is a symmetric matrix, and we use only its lower part (if uplo='L') or upper part (if uplo='U')
  template<class T>
  inline void blas_symm(char uplo, T alpha, const BMat<T>& A, const BMat<T>& B, T beta, const BMat<T>& C)
  { 
    assert( uplo=='L' || uplo=='U' );
    assert(A.nrows()==A.ncols() && A.ncols()==B.nrows() && B.nrows()==C.nrows() && B.ncols()==C.ncols());
    char leftside = 'L'; 
    blas_symm(leftside, uplo, C.nrows(), C.ncols(), 
              alpha, A.data(), A.stride(), B.data(), B.stride(),
              beta, C.data(), C.stride()); 
  }

template<class T, class Talpha, class Tbeta>
inline void blas_symm(char uplo, Talpha alpha, const BMat<T>& A, const BMat<T>& B, Tbeta beta, const BMat<T>& C)
  { blas_symm(uplo, T(alpha), A, B, T(beta), C); }

  //! C = alpha B A + beta C 
  //! A is a symmetric matrix, and we use only its lower part (if uplo='L') or upper part (if uplo='U')
  template<class T>
  inline void blas_symm_rightside(char uplo, T alpha, const BMat<T>& A, const BMat<T>& B, T beta, const BMat<T>& C)
  { 
    assert( uplo=='L' || uplo=='U' );
    assert(A.nrows()==A.ncols() && A.nrows()==B.ncols() && B.nrows()==C.nrows() && B.ncols()==C.ncols());
    char rightside = 'R'; 
    blas_symm(rightside, uplo, C.nrows(), C.ncols(), 
              alpha, A.data(), A.stride(), B.data(), B.stride(), beta, C.data(), C.stride()); 
  }

template<class T, class Talpha, class Tbeta>
inline void blas_symm_rightside(char uplo, Talpha alpha, const BMat<T>& A, const BMat<T>& B, Tbeta beta, const BMat<T>& C)
  { blas_symm_rightside(uplo, T(alpha), A, B, T(beta), C); }


  //! C = alpha A A' + beta C 
  //! C is a symmetric matrix, and we use only its lower part (if uplo='L') or upper part (if uplo='U')
  template<class T>
  inline void blas_syrk(char uplo, T alpha, const BMat<T>& A, T beta, const BMat<T>& C)
  { 
    assert( uplo=='L' || uplo=='U' );
    assert(C.nrows()==C.ncols() && A.nrows()==C.nrows()); 
    blas_syrk(uplo, 'N', A.nrows(), A.ncols(), 
              alpha, A.data(), A.stride(), beta, C.data(), C.stride() ); 
  }

  template<class T, class Talpha, class Tbeta>
  inline void blas_syrk(char uplo, Talpha alpha, const BMat<T>& A, Tbeta beta, const BMat<T>& C)
  { blas_syrk(uplo, T(alpha), A, T(beta), C); }


  //! C = alpha A' A + beta C 
  //! C is a symmetric matrix, and we consider only its lower part (if uplo='L') or upper part (if uplo='U') */
  template<class T>
  inline void blas_syrk_T(char uplo, T alpha, const BMat<T>& A, T beta, const BMat<T>& C)
  { 
    assert( uplo=='L' || uplo=='U' );
    assert(C.nrows()==C.ncols() && A.ncols()==C.nrows()); 
    blas_syrk(uplo, 'T', A.ncols(), A.nrows(), 
              alpha, A.data(), A.stride(), beta, C.data(), C.stride() ); 
  }

  template<class T, class Talpha, class Tbeta>
  inline void blas_syrk_T(char uplo, Talpha alpha, const BMat<T>& A, Tbeta beta, const BMat<T>& C)
  { blas_syrk_T(uplo, T(alpha), A, T(beta), C); }

  //! C = alpha A B' + B A' + beta C 
  //! C is a symmetric matrix, and we use only its lower part (if uplo='L') or upper part (if uplo='U') 
  template<class T>
  inline void blas_syr2k(char uplo, T alpha, const BMat<T>& A, const BMat<T>& B, T beta, const BMat<T>& C)
  { 
    assert( uplo=='L' || uplo=='U' );    
    assert(C.nrows()==C.ncols() && A.nrows()==C.nrows() && B.nrows()==C.nrows() && A.ncols()==B.ncols());
    blas_syr2k(uplo, 'N', A.nrows(), A.ncols(), 
               alpha, A.data(), A.stride(), B.data(), B.stride(), beta, C.data(), C.stride() ); 
  }

  template<class T, class Talpha, class Tbeta>
  inline void blas_syr2k(char uplo, Talpha alpha, const BMat<T>& A, const BMat<T>& B, Tbeta beta, const BMat<T>& C)
  { blas_syr2k(uplo, T(alpha), A, B, T(beta), C); }

  //! C = alpha A' B + B' A + beta C 
  //! C is a symmetric matrix, and we use only its lower part (if uplo='L') or upper part (if uplo='U')
  template<class T>
  inline void blas_syr2k_T(char uplo, T alpha, const BMat<T>& A, const BMat<T>& B, T beta, const BMat<T>& C)
  { 
    assert( uplo=='L' || uplo=='U' );
    int n = C.nrows();
    assert(C.ncols()==n && A.ncols()==n && B.ncols()==n && A.nrows()==B.nrows()); 
    blas_syr2k(uplo, 'T', A.ncols(), A.nrows(), 
               alpha, A.data(), A.stride(), B.data(), B.stride(), beta, C.data(), C.stride() ); 
  }

  template<class T, class Talpha, class Tbeta>
  inline void blas_syr2k_T(char uplo, Talpha alpha, const BMat<T>& A, const BMat<T>& B, Tbeta beta, const BMat<T>& C)
  { blas_syr2k_T(uplo, T(alpha), A, B, T(beta), C); }



  // ------------------------------------
  // Generic Linear algebra operations

//! returns the sum of the elements of vector x
template<class T> 
inline T sum(const BVec<T>& x)
{
    T s = 0;
    const T* ptr = x.data();
    for(int i=0; i<x.length(); i++, ptr += x.step())
        s += *ptr;
    return s;
}

//! returns the product of the elements of vector x
template<class T> 
inline T product(const BVec<T>& x)
{
    T s = 1;
    const T* ptr = x.data();
    for(int i=0; i<x.length(); i++, ptr += x.step())
        s *= *ptr;
    return s;
}


//! returns the index (0-based) of the element with the largest absolute value
template<class T> 
inline int argmax_abs(const BVec<T>& x)
{
    const T* ptr = x.data();
    int n = x.length();
    int step = x.step();
    int max_pos = 0;
    T max_abs_val = myabs(*ptr);
    ptr += step;
    for(int i=1; i<n; i++, ptr += step)
    {
        T abs_val = myabs(*ptr);
        if (abs_val > max_abs_val)
        {
            max_abs_val = abs_val;
            max_pos = i;
        }
    }
    return max_pos;
}



//! inverts each element of x i.e. x[i] = 1/x[i]
template<class T> 
inline void invert_elements(const BVec<T>& x)
{
    T* ptr = x.data();
    for(int i=0; i<x.length(); i++, ptr += x.step())
        *ptr = 1/(*ptr);
}

//! takes the square root of each element of x i.e. x[i] = sqrt(x[i])
template<class T> 
inline void sqrt_elements(const BVec<T>& x)
{
    T* ptr = x.data();
    for(int i=0; i<x.length(); i++, ptr += x.step())
        *ptr = sqrt(*ptr);
}


  //! z = x-y
  template<class T> 
  inline void diff(const BVec<T>& x, const BVec<T>& y, const BVec<T>& z)
  {
    z << x;
    blas_axpy(-1, y, z);
  }

  //! C = A-B
  template<class T> 
  inline void diff(const BMat<T>& A, const BMat<T>& B, const BMat<T>& C)
  {
    C << A;
    blas_axpy(-1, B, C);
  }


//! C = C+AB  where all matrices use dense representation, but B is mostly zeros (sparse).
//! The same result can be obtained by calling gemm, but here we avoid unnecessary
//! computation due to the many elements of B that are 0.
template<class T>
void accumulate_product_with_densely_stored_sparse_mat(const BMat<T>& A, const BMat<T>& B, const BMat<T>& C)
{
    assert(A.nrows()==C.nrows() && B.ncols()==C.ncols() && A.ncols()==B.nrows());
    for (int j=0; j<B.ncols(); j++)
    {
        BVec<T> Bj = B.column(j);
        BVec<T> Cj = C.column(j);
        for(int i=0; i<B.nrows(); i++)
        {
            T Bij = Bj[i];
            if (Bij!=0)
                blas_axpy( Bij, A.column(i), Cj );
        }
    }
}

template<class T>
void product_with_densely_stored_sparse_mat(const BMat<T>& A, const BMat<T>& B, const BMat<T>& C)
{
    C.clear();
    accumulate_product_with_densely_stored_sparse_mat(A, B, C);
}

  //! diag(A) <- diag(A) + alpha
  template<class T>
  inline void add_scalar_to_diagonal(T alpha, const BMat<T>& A) 
  { 
    int d = (A.nrows()<A.ncols() ?A.nrows() :A.ncols()); 

    // Generic version
    T* ptr = A.data();
    int step = A.stride()+1;
    for( int i=0; i<d; i++, ptr += step)
      *ptr += alpha;

    // Blas version (possibly more efficient, but using funny stepsize of 0 possibly not supported everywhere)
    /*
      T one = 1;
      blas_axpy(d, alpha, &one, 0, A.data(), A.stride()+1); 
    */
  }

  template<class T, class Talpha>
  inline void add_scalar_to_diagonal(Talpha alpha, const BMat<T>& A) 
  { add_scalar_to_diagonal(T(alpha), A); } 


  template<class T>
  void transpose_squaremat_inplace(const BMat<T>& A)
  {
    assert(A.nrows()==A.ncols());

    // slow unefficient version
    for(int i=1; i<A.nrows(); i++)
      for(int j=0; j<i; j++)
        {
          T tmp = A(i,j);
          A(i,j) = A(j,i);
          A(j,i) = tmp;
        }
  }

  //! Copies the lower part of a square matrix (below the diagonal) to the upper part. Thus making it a symmetric matrix.
  template<class T>
  void copy_lower_to_upper(const BMat<T>& A)
  {
    assert(A.nrows()==A.ncols());

    // slow unefficient version
    for(int j=0; j<A.ncols()-1; j++)
      for(int i=j+1; i<A.nrows(); i++)
        A(j,i) = A(i,j);
  }

  //! Copies the upper part of a square matrix (above the diagonal) to the lower part. Thus making it a symmetric matrix.
  template<class T>
  void copy_upper_to_lower(const BMat<T>& A)
  {
    assert(A.nrows()==A.ncols());

    // slow unefficient version
    for(int j=0; j<A.ncols()-1; j++)
      for(int i=j+1; i<A.nrows(); i++)
        A(i,j) = A(j,i);
  }

template<class T>
T min(const BMat<T>& A)
{    
    T minval = A(0,0);
    for(int j=0; j<A.ncols(); j++)
    {
        for(int i=0; i<A.nrows(); i++)
        {
            T a = A(i,j);
            if (a<minval)
                minval = a;
        }
    }
    return minval;
}

template<class T>
T max(const BMat<T>& A)
{    
    T maxval = A(0,0);
    for(int j=0; j<A.ncols(); j++)
    {
        for(int i=0; i<A.nrows(); i++)
        {
            T a = A(i,j);
            if (a>maxval)
                maxval = a;
        }
    }
    return maxval;
}

  template<class T>
  T max_abs(const BMat<T>& A)
  {    
    T maxabs = 0;
    for(int j=0; j<A.ncols(); j++)
      {
        for(int i=0; i<A.nrows(); i++)
          {
            T a = myabs(A(i,j));
            if (a>maxabs)
              maxabs = a;
          }
      }
    return maxabs;
  }

  template<class T>
  T max_abs_diff(const BVec<T>& x, const BVec<T>& y)
  {
      assert(x.length() == y.length());
    
    T maxdiff = 0;
    for(int i=0; i<x.length(); i++)
      {
        T ad = myabs(x[i] - y[i]);
        if (ad>maxdiff)
          maxdiff = ad;
      }
    return maxdiff;
  }

  template<class T>
  T max_abs_diff(const BMat<T>& A, const BMat<T>& B)
  {
    assert(A.nrows()==B.nrows() && A.ncols()==B.ncols());
    
    T maxdiff = 0;
    for(int j=0; j<A.ncols(); j++)
      {
        for(int i=0; i<A.nrows(); i++)
          {
            T ad = myabs(A(i,j)-B(i,j));
            if (ad>maxdiff)
              maxdiff = ad;
          }
      }
    return maxdiff;
  }

  //!  A = A diag(x)  
  template<class T>
  inline void scale_columns(const BMat<T>& A, const BVec<T>& scales) 
  { 
      int n = A.nrows();
      assert(scales.length() == A.ncols());
      T* Ak = A.data();
      for(int k=0; k<A.ncols(); k++, Ak += A.stride())       
          blas_scal(n, scales[k], Ak, 1); 
  } 

  //!  B = A diag(x)  
  template<class T>
  inline void scale_columns(const BMat<T>& A, const BVec<T>& scales, const BMat<T>& B) 
  { 
      int nrows = A.nrows();
      int ncols = A.ncols();      
      assert(scales.length() == A.ncols());
      T* Ak = A.data();
      T* Bk = B.data();
      for(int k=0; k<ncols; k++, Ak += A.stride(), Bk += B.stride() )       
      {
          T s = scales[k];
          for (int i=0; i<nrows; i++)
              Bk[i] = Ak[i]*s;
      }
  } 


  //!  A = diag(x) A  
  template<class T>
  inline void scale_rows(const BMat<T>& A, const BVec<T>& scales) 
  { 
      assert(scales.length() == A.nrows());    
    T* Aj = A.data();
    for(int j=0; j<A.ncols(); j++, Aj += A.stride())
        for(int i=0; i<A.nrows(); i++)
            Aj[i] *= scales[i];
  } 


  //!  B = diag(x) A  
  template<class T>
  inline void scale_rows(const BMat<T>& A, const BVec<T>& scales, const BMat<T>& B) 
  { 
      int nrows = A.nrows();
      int ncols = A.ncols();      
      assert(scales.length() == nrows);    

      T* Ak = A.data();
      T* Bk = B.data();
      int A_stride = A.stride();
      int B_stride = B.stride();
      for(int k=0; k<ncols; k++, Ak += A_stride, Bk += B_stride )       
      {
          for (int i=0; i<nrows; i++)
              Bk[i] = Ak[i]*scales[i];
      }
  } 




  //!  C = A diag(x)  
  template<class T>
  inline void product_matrix_diagonal(const BMat<T>& A, const BVec<T>& x, const BMat<T>& C) 
  {   
    C << A;
    scale_columns(C,x);
  } 


//! for each row of the matrix, computes the sum of its elements
template<class T> 
inline void sum_rowwise(const BMat<T>& A, const BVec<T>& x)
{
    int nrows = A.nrows();
    int ncols = A.ncols();
    assert(x.length()==nrows);
    if (ncols==0)
        return;

    x << A.column(0);
    for(int j=1; j<ncols; j++)
        x += A.column(j);
}

//! for each column of the matrix A, computes the sum of its elements
template<class T> 
inline void sum_columnwise(const BMat<T>& A, const BVec<T>& x)
{
    // int nrows = A.nrows();
    int ncols = A.ncols();
    assert(x.length()==ncols);
    for(int j=0; j<ncols; j++)
        x[j] = sum(A.column(j));
}

  //! Returns sum( A * B )  where * denotes elementwise product and sum is the sum over all elements.
  template<class T>
  inline T sum_prod(const BMat<T>& A, const BMat<T>& B)
  {
    T res = 0;
    if( A.is_contiguous() && B.is_contiguous() )
      {
        res = blas_dot(A.flat(), B.flat());
      }
    else
      {
        for(int j=0; j<A.ncols(); j++)
          res += blas_dot(A.column(j), B.column(j) );
      }
    return res;
  }

  template<class T>
  inline T trace(const BMat<T>& A)
  {
    assert(A.nrows()==A.ncols()); // trace is usually for squared matrices
    T res = 0;
    for (int i=0; i<A.nrows(); i++)
      res += A(i,i);
    return res;
  }


// My own linalg functions


//! Will apply the power iteration method to compute the largest eigenvalue
//! and associated eigenvector.
//! You should provide a guess for the initial eigenvector x (or a random
//! initialized vector)
//! The call will modify x which should converge to the leading eigenvector
//! (and be of norm 1).
//! The call will return the associated eigenvalue.
//! The call will actually perform twice the number of specified iterations.
template<class T>
T power_iteration(const BMat<T>& A, const BVec<T>& x, int niter=10)
{
    static BVec<T> other_x;

    int d = x.length();
    assert(A.ncols() == d);
    assert(A.nrows() == A.ncols());

    other_x.resize(d);
    T coef;
    for (int i=0; i<niter; i++)
    {
        coef = (T)(1/sqrt(blas_dot(x,x)));
        blas_gemv('N', coef, A, x, 0, other_x);
        coef = (T)(1/sqrt(blas_dot(other_x,other_x)));
        blas_gemv('N', coef, A, other_x, 0, x);        
    }
    // renormalize x
    coef = (T)(1/sqrt(blas_dot(x,x)));
    x *= coef;

    // compute the eigenvalue. We have A x = lambda x
    blas_gemv('N', 1, A, x, 0, other_x);
    int pos = argmax_abs(other_x);
    T lambda = other_x[pos]/x[pos];
    return lambda;
}


//! Will apply the power iteration method to compute the largest singular value 
//! and associated left and right singular vectors of m x n matrix A.
//! left_singvec is a vector of size m, right_singvec is of size n.
//! left_singvec should be initialized to a vector of ones, or to a guess of
//! the leading left singular vector.
//! right_singvec needs not be initialized in any particular way (but must have
//! size n). 
//! The call will return the associated leading singular value.
//!
//! The leading left singular vector of A is obtained through power iteration 
//! as the leading eigenvector of A A^T, and the leading
//! eigenvector of A^T A yields the right singular vector, and is obtained
//! during the same power iteration.  
template<class T>
T power_iteration_for_singular_vect(const BMat<T>& A, const BVec<T>& left_singvec, const BVec<T>& right_singvec, int niter=100)
{
    assert(left_singvec.length() == A.nrows());
    assert(right_singvec.length() == A.ncols() );

    T coef;
    for (int i=0; i<niter; i++)
    {
        coef = (T)(1/sqrt(blas_dot(left_singvec,left_singvec)));
        blas_gemv('T', coef, A, left_singvec, 0, right_singvec);
        blas_gemv('N', 1, A, right_singvec, 0, left_singvec);        
    }

    // renormalize left_singvec before estimating eigenvalue (for possibly improved numerical precision)
    left_singvec *= (T)(1/sqrt(blas_dot(left_singvec,left_singvec)));

    // compute the eigenvalue of A A^T after assumed convergence: we have A A^T left_singvec = lambda left_singvec
    int pos = argmax_abs(left_singvec);
    T initial_x_val = left_singvec[pos];
    blas_gemv('T', 1, A, left_singvec, 0, right_singvec);
    blas_gemv('N', 1, A, right_singvec, 0, left_singvec);        
    T final_x_val = left_singvec[pos];
    T eigval = final_x_val / initial_x_val;
    // singular value is square root of absolute eigenvalue
    T singval = sqrt(myabs(eigval));

    // finally renormalize singular vectors
    left_singvec *= (T)(1/sqrt(blas_dot(left_singvec,left_singvec)));
    right_singvec *= (T)(1/sqrt(blas_dot(right_singvec,right_singvec)));

    return singval;
}




  // -------------------------
  // Lapack Linear algebra operations




/*!   Solves AX = B
  This is a simple wrapper over the lapack routine. It A and B as input, 
  as well as storage for resulting pivots vector of ints of same length as A has rows.
  The call overwrites B, putting the solution X in there,
  and A is also overwritten to contain the factors L and U from the factorization A = P*L*U; 
  (the unit diagonal elements of L  are  not stored).
  The lapack status is returned:
  = 0:  successful exit
  < 0:  if INFO = -i, the i-th argument had an illegal value
  > 0:  if INFO = i, U(i,i) is  exactly  zero.   The factorization has been completed, 
  but the factor U is exactly singular, so the solution could not be computed.
*/

template<class T>
inline int lapackSolveLinearSystem(const BMat<T>& A, const BMat<T>& B, const BVec<int>& pivots)
{
    assert(A.nrows() == A.ncols());
    assert(A.nrows() == B.nrows());
    assert(pivots.length()==A.nrows());
    int INFO;
    int N = A.nrows();
    int NRHS = B.ncols();
    T* Aptr = A.data();
    int LDA = A.stride();
    int* IPIVptr = pivots.data();
    T* Bptr = B.data();
    int LDB = B.stride();
    lapack_Xgesv_(&N, &NRHS, Aptr, &LDA, IPIVptr, Bptr, &LDB, &INFO);
    return INFO;
}

//! Utility call that solves for X the system of linear equations AX=B with a square matrix A
//! It calls the lower level lapackSolveLinearSystem but does the necessary
//! copies so that A and B are not overwritten. 

template<class T>
inline void solveLinearSystem(const BMat<T>& A, BMat<T>& X, const BMat<T>& B)
{
    static BVec<int> pivots;
    static BMat<T> Acopy;

    assert(A.nrows() == A.ncols());
    assert(A.nrows() == B.nrows());
    pivots.resize(A.nrows());
    Acopy.resize(A.nrows(), A.ncols());
    Acopy << A;
    X.resize(B.nrows(), B.ncols());
    X << B;

    int info = lapackSolveLinearSystem(Acopy, X, pivots);
    if (info!=0)
        PLERROR("lapackSolveLinearSystem returned with error info status of " << info);
}

//! This function inverts a matrix in place.

int lapackInvertMatrix(const BMat<real>& A)
{
    // If the matrix is empty, just do nothing instead of crashing.
    if (A.isEmpty()) {
        return 0;
    }

    // some checks

    int M = A.nrows();
    int N = A.ncols();
    if (M != N)
        PLERROR("The input matrix must be square!");
    real* Adata = A.data();
    int LDA = A.stride();
    static BVec<int> ipiv;
    ipiv.resize(N);
    int INFO;

    lapack_Xgetrf_(&M, &N, Adata, &LDA, ipiv.data(), &INFO);

    if (INFO != 0)
    {
        cout << "In matInvert: Error doing the inversion." << endl
             << "Check the man page of <sgetrf> with error code " << INFO
             << " for more details." << endl;

        return INFO;
    }

    int LWORK = N;
    static BVec<real> work;
    work.resize(LWORK);

    lapack_Xgetri_(&N, Adata, &LDA, ipiv.data(), work.data(), &LWORK, &INFO);

    if (INFO != 0)
    {
        cout << "In matInvert: Error doing the inversion." << endl
             << "Check the man page of <sgetri> with error code " << INFO
             << " for more details." << endl;

        return INFO;
    }

    return INFO;
}


void invertMatrix(const BMat<real>& A)
{
    lapackInvertMatrix(A);
}

/*! Performs the SVD decomposition A = U.S.Vt
(code adapted from old PLearn)

  This is a straight forward call to the lapack function.
  CAREFUL: the A matrix argument is changed in the process!

  Note that U S and V are resized by the call (if you did not already provide them with the right size)
*/

// (will work for float and double)

template<class num_t>
void lapackSVD(const BMat<num_t>& A, BMat<num_t>& U, BVec<num_t>& S, BMat<num_t>& VT, char JOBZ='A', double safeguard = 1)
{            
    int M = A.nrows();
    int N = A.ncols();
    int LDA = A.stride();
    int min_M_N = min(M,N);
    S.resize(min_M_N);

    assert(S.is_contiguous());

    switch(JOBZ)
    {
    case 'A':
        U.resize(M,M);
        VT.resize(N,N);
        break;
    case 'S':
        U.resize(min_M_N, M);
        VT.resize(N, min_M_N);
        break;
    case 'O':
        if(M<N)
            U.resize(M,M); // and VT is not used      
        else
            VT.resize(N,N); // and U is not used
        break;
    case 'N':
        break;
    default:
        PLERROR("In lapackSVD, bad JOBZ argument : " << JOBZ);
    }


    int LDU = 1;
    int LDVT = 1;
    num_t* Udata = 0;
    num_t* VTdata = 0;

    if(VT.isNotEmpty())
    {
        LDVT = VT.stride();
        VTdata = VT.data();
    }
    if(U.isNotEmpty())
    {
        LDU = U.stride();
        Udata = U.data();
    }

    static BVec<num_t> WORK;
    WORK.resize(1);
    int LWORK = -1;

    static BVec<int> IWORK;
    IWORK.resize(8*min_M_N);

    int INFO;

    // first call to find optimal work size
    lapack_Xgesdd_(&JOBZ, &M, &N, A.data(), &LDA, S.data(), Udata, &LDU, VTdata, &LDVT, WORK.data(), &LWORK, IWORK.data(), &INFO);

    if(INFO!=0)
    {
        PLERROR("In lapackSVD, problem in first call to sgesdd_ to get optimal work size, returned INFO = " << INFO); 
    }
  
    // make sure we have enough space
    LWORK = int(WORK[0] * safeguard + 0.5); // optimal size (safeguard may be used to make sure it doesn't crash in some rare occasions).
    WORK.resize(LWORK);
    // cerr << "Optimal WORK size: " << LWORK << endl;

    // second call to do the computation
    lapack_Xgesdd_(&JOBZ, &M, &N, A.data(), &LDA, S.data(), Udata, &LDU, VTdata, &LDVT, WORK.data(), &LWORK, IWORK.data(), &INFO );

    if(INFO!=0)
    {      
        // cerr << A << endl;
        // cerr << "In lapackSVD, failed with INFO = " << INFO << endl;
        PLERROR("In lapackSVD, problem when calling sgesdd_ to perform computation, returned INFO = " << INFO);
    }
}

// --------------------------------------------------
// Sherman-Morrison and Woodbury based low rank updates of inverse transposed


// Performs rank one update to U^-T (inverse of U transposed) that corresponds to rank 1 update to U <- U + alpha u v^T
// using Sherman-Morrison formula.
template<typename Mat, typename Vec>
void blas_rank_1_update_UinvT(const Mat& UinvT, typename Mat::elem_t alpha,
                              const Vec& u, const Vec& v) 
{
    // d-dimensional vectors for intermediate computations
    typedef typename Vec::elem_t elem_t;
    static Vec u_tilde;  
    static Vec v_tilde;  
    
    
    int d = UinvT.nrows();
    u_tilde.resize(d);
    v_tilde.resize(d);

    // u_tilde = UinvT^T u
    blas_gemv('T', 1, UinvT, u, 0, u_tilde);
    // v_tilde = UinvT v
    blas_gemv('N', 1, UinvT, v, 0, v_tilde);
    elem_t s = blas_dot(v, u_tilde);
    elem_t alpha_tilde = -alpha/(1+alpha*s);
    // UinvT = UinvT + alpha_tilde v_tilde u_tilde^T
    blas_ger(alpha_tilde, v_tilde, u_tilde, UinvT); 
}

// Performs rank k update to U^-T (inverse of U transposed) that corresponds to rank k update to U <- U + alpha A B^T
// where U is a d x d matrix,  A and B are d x k matrices and alpha is a scalar. 
// This is done by calling k times rank-one updates
template<typename Mat>
void blas_rank_update_UinvT(const Mat& UinvT, typename Mat::elem_t alpha, const Mat& A, const Mat& B)  
{
    for (int k=0; k<A.ncols(); k++)
        blas_rank_1_update_UinvT(UinvT, alpha, A.column(k), B.column(k));
}

// rankm_update_invT rank-m update to square matrix and corresponding update to its inverse.
// Performs:
// U <- U + alpha A B^T and correspondingly updates UinvT = U^-T
// U is a d x d matrix, UinvT is its inverse transposed
// A and B are d x k matrices.
// inv_update_mode specifieds how to perform the invere update: 1: iterate Sherman-Morrison rank-1 updates; 2: use Woodbury identity; 3: recompute full inverse
// All matrices must be in column major mode

template<typename Mat>
void rankm_update_U_and_UinvT_iter_v1(typename Mat::elem_t alpha, const Mat& A, const Mat& B, const Mat& U, const Mat& UinvT)  
  
{
    // 7) U <- U + alpha A B^T
    blas_gemm_NT(alpha, A, B, 1, U);

    // 8) Corresponding update to UinvT
    blas_rank_update_UinvT(UinvT, alpha, A, B);
}


// rank_update_invT rank-m update to square matrix and corresponding update to its inverse.
// Performs:
// U <- U + alpha A B^T and correspondingly updates UinvT = U^-T
// U is a d x d matrix, UinvT is its inverse transposed
// A and B are d x K matrices.
// inv_update_mode specifieds how to perform the invere update: 1: iterate Sherman-Morrison rank-1 updates; 2: use Woodbury identity; 3: recompute full inverse
// All matrices must be in column major mode

// To be thoroughly checked
// This version also has a special case treatment for the first iteration
template<typename Mat>
void rankm_update_U_and_UinvT_iter_v2(typename Mat::elem_t alpha, const Mat& A, const Mat& B, const Mat& U, const Mat& UinvT)
{
    typedef typename Mat::vec_t Vec;
    typedef typename Mat::elem_t elem_t;

    assert(A.nrows()==B.nrows() && A.ncols()==B.ncols());

    // d-dimensional vectors for intermediate results
    static Vec u;
    static Vec v;

    int d = U.nrows();
    u.resize(d);
    v.resize(d);

    // 7) U <- U + alpha A B^T
    blas_gemm_NT(alpha, A, B, 1, U);

    // 8) Corresponding update to UinvT

    for (int k=0; k<B.ncols(); k++)
    {
        Vec Bk = B.column(k);
        blas_gemv('N', 1, UinvT, Bk, 0, u);

        if (k==0)
        {
            elem_t Bk_v = blas_dot(Bk,Bk);          
            elem_t scale = -alpha / (1+alpha*Bk_v);
            blas_ger(scale, u, Bk, UinvT);
        }
        else
        {
            Vec Ak = A.column(k);          
            blas_gemv('T', 1, UinvT, Ak, 0, v);
            elem_t Bk_v = blas_dot(Bk, v);
            elem_t scale = -alpha / (1+alpha*Bk_v);
            blas_ger(scale, u, v, UinvT);
        }
    }
}

// rank-m update to square matrix and to its inverse transpose, using the Woodbury identity
// Updates U and UinvT with the following U <- U + alpha A B^T  
// Based on Woodubry identity (internally performs inverse of a m x m matrix )
// B is a d x m matrix
void OLD_BUGGY_rankm_update_U_and_UinvT_Woodbury(real alpha, const BMat<real>& A, const BMat<real>& B, const BMat<real>& U, const BMat<real>& UinvT)
{
    int d = B.nrows();
    int m = B.ncols();

    // if(m>d)
    //    PLERROR("It makes no sense to update the inverse using Woodbury identity. It will be cheaper to compute the inverse of your updated U directly");


    // 7) U <- U + alpha A B^T
    blas_gemm_NT(alpha, A, B, 1, U);

    // 8) Corresponding update to UinvT  ( V is B^T  U is A    A is U  ) 
    // U^-T <- U^-T - U^-T B (1/alpha I + B^T U^-1 A)^-T A^T U^-T
    // or alternatively: U^-T <- U^-T - U^-T B (A^T U^-T B + 1/alpha I)^-1 A^T U^-T
    
    // Compute Imm = ( B^T B + 1/alpha I )^-1
    // Note TODO: we could (and probably should) compute and invert a symmetric Imm (using only its lower or upper part)
    static BMat<real> Imm;
    Imm.resize(m,m);
    blas_gemm_TN(1, B, B, 0, Imm);
    add_scalar_to_diagonal(1/alpha, Imm);
    invertMatrix(Imm);    

    // Compute B_Imm = B Imm   (a d x m matrix )
    static BMat<real> B_Imm;
    B_Imm.resize(d,m);
    blas_gemm_NN(1, B, Imm, 0, B_Imm);

    // Compute UinvT_B_Imm  (a d x m matrix )
    static BMat<real> UinvT_B_Imm;
    UinvT_B_Imm.resize(d,m);
    blas_gemm_NN(1, UinvT, B_Imm, 0, UinvT_B_Imm);
    
    // Perform update UinvT <- UinvT - UinvT_B_Imm B^T
    blas_gemm_NT(-1, UinvT_B_Imm, B, 1, UinvT);
}



// rank-m update to square matrix and to its inverse transpose, using the Woodbury identity
// Updates U and UinvT with the following U <- U + alpha A B^T  
// Based on Woodubry identity (internally performs inverse of a m x m matrix )
// B is a d x m matrix
// Note: last parameter Uinv_A can be passed to save some computations
// (note that it is Uinv_A not UinvT_A) as it can be trivially available in some cases.
// If not, call the version of that function without that
// parameter (version further down), that will compte it.
template<class Mat>
void rankm_update_U_and_UinvT_Woodbury(typename Mat::elem_t alpha, const Mat& A, const Mat& B, const Mat& U, const Mat& UinvT, const Mat& optional_Uinv_A)
{
    int d = B.nrows();
    int m = B.ncols();

    // if(m>d)
    //    PLERROR("It makes no sense to update the inverse using Woodbury identity. It will be cheaper to compute the inverse of your updated U directly");


    // 7) U <- U + alpha A B^T
    blas_gemm_NT(alpha, A, B, 1, U);

    // 8) Corresponding update to UinvT  ( V is B^T  U is A    A is U  ) 
    // U^-T <- U^-T - U^-T B (1/alpha I + B^T U^-1 A)^-T A^T U^-T
    // or alternatively: U^-T <- U^-T - U^-T B (A^T U^-T B + 1/alpha I)^-1 A^T U^-T
    // or alternatively: U^-T <- U^-T - U^-T B ( (U^-1 A)^T B + 1/alpha I)^-1 (U^-1 A)^T
    //                   U^-T <- U^-T - U^-T B ( (Uinv_A)^T B + 1/alpha I)^-1 (Uinv_A)^T

    // this was called with A=H_tilde=UH so that Uinv_A = U^-1 U H = H
    
    // Compute Uinv_A = U^-1 A is not provided
    const Mat& Uinv_A = optional_Uinv_A;

// Compute Imm = ( (Uinv_A)^T B + 1/alpha I )^-1
    static Mat Imm;
    Imm.resize(m,m);
    blas_gemm_TN(1, Uinv_A, B, 0, Imm);
    add_scalar_to_diagonal(1/alpha, Imm);
    invertMatrix(Imm);    

    // Compute B_Imm = B Imm   (a d x m matrix )
    static Mat B_Imm;
    B_Imm.resize(d,m);
    blas_gemm_NN(1, B, Imm, 0, B_Imm);

    // Compute UinvT_B_Imm  (a d x m matrix )
    static Mat UinvT_B_Imm;
    UinvT_B_Imm.resize(d,m);
    blas_gemm_NN(1, UinvT, B_Imm, 0, UinvT_B_Imm);
    
    // Perform update UinvT <- UinvT - UinvT_B_Imm (Uinv_A)^T
    blas_gemm_NT(-1, UinvT_B_Imm, Uinv_A, 1, UinvT);
}

template<class Mat>
void rankm_update_U_and_UinvT_Woodbury(typename Mat::elem_t alpha, const Mat& A, const Mat& B, const Mat& U, const Mat& UinvT)
// rank-m update to square matrix and to its inverse transpose, using the Woodbury identity
// Updates U and UinvT with the following U <- U + alpha A B^T  
// Based on Woodubry identity (internally performs inverse of a m x m matrix )
// B is a d x m matrix
// NOTE: If you can cheaply provide (U^-1 A) consider drectly calling the version of
// this funciton that takes it (Uinv_A) as extra parameter
// to avoid recomputing it (which the present call does).
{    
    static Mat Uinv_A;
    Uinv_A.resize(UinvT.ncols(), A.ncols());
    blas_gemm_TN(1, UinvT, A, 0, Uinv_A);
    rankm_update_U_and_UinvT_Woodbury(alpha, A, B, U, UinvT, Uinv_A);
}

// rankm_update_U_and_UinvT_recompute rank-m update to square matrix and recomputes its inverse from scratch.
// Performs:
// U <- U + alpha A B^T and correspondingly updates UinvT = U^-T
// U is a d x d matrix, UinvT is its inverse transposed
// A and B are d x k matrices.
// All matrices must be in column major mode

template<class Mat>
void rankm_update_U_and_UinvT_recompute(typename Mat::elem_t alpha, const Mat& A, const Mat& B, const Mat& U, const Mat& UinvT)  // how big should this be????
{
    // 7) U <- U + alpha A B^T
    blas_gemm_NT(alpha, A, B, 1, U);

    // 8) Corresponding update to UinvT
    // We recompute the inverse of U from scratch
    UinvT << U;
    transpose_squaremat_inplace(UinvT);
    invertMatrix(UinvT);
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
