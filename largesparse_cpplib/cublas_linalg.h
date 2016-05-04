// -*- C++ -*-

// cublas_linalg.h
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


#ifndef cublas_linalg_INC
#define cublas_linalg_INC

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <math.h>



// #define real float
#include "pl_math.h"

#include "plerror.h"

#include "CudaStorage.h"
#include "cublas_v2.h"

#include "blas_linalg.h"

namespace PLearn {

using std::max;
using std::min;


inline static void check_cuda_error(cudaError_t err, const string& doingwhat="") 
{
    if (err != cudaSuccess) 
        PLERROR("CUDA operation failed: " << doingwhat << " " << cudaGetErrorString(err) );
}


//! Converts 'N' -> CUBLAS_OP_N
inline cublasOperation_t toCublasOp(char char_op)
{ 
    cublasOperation_t cublas_op = CUBLAS_OP_N;
    switch(char_op)
    {
    case 'N':
        cublas_op = CUBLAS_OP_N;
        break;

    case 'T':
        cublas_op = CUBLAS_OP_T;
        break;

    case 'C':
        cublas_op = CUBLAS_OP_C;
        break;

    default:
        PLERROR("Invalid operation char");
    }
    return cublas_op;
}


//! convert 'L' -> CUBLAS_FILL_MODE_LOWER  'U'  -> CUBLAS_FILL_MODE_UPPER
inline cublasFillMode_t toCublasFillMode(char char_fill_mode)
{ 
    assert(char_fill_mode=='L' || char_fill_mode=='U');
    return char_fill_mode=='L' ?CUBLAS_FILL_MODE_LOWER :CUBLAS_FILL_MODE_UPPER;
}


class CudaStreamArray
{
protected:
    mutable cudaStream_t *streams;
    mutable int length_;

public:

    inline CudaStreamArray()
        :streams(0), length_(0)
    {}

    inline CudaStreamArray(int n)
         :streams(new cudaStream_t[n]),
         length_(n)
    {
        for (int i=0; i<length_; i++)
            check_cuda_error(cudaStreamCreate(&streams[i]),
                             "cudaCreateStream");
    }

    void synchronize() const
    {
        for (int k=0; k<length_; k++)
            cudaStreamSynchronize(streams[k]);  
    }

    inline void free()
    {
        if (streams!=0)
        {
            for (int i=0; i<length_; i++)
                check_cuda_error(cudaStreamDestroy(streams[i]),
                                 "cudaStreamDestroy");
            streams = 0;
            length_ = 0;
        }
    }

    inline void operator=(const CudaStreamArray& other)
    {
        streams = other.streams;
        length_ = other.length_;
        other.streams = 0;  // steal the other's streams array
        other.length_ = 0;
    }

    inline CudaStreamArray(const CudaStreamArray& other)
        :streams(0), length_(0)
    { 
        operator=(other);
    }


    inline int length() const
    { return length_; }

    
    inline cudaStream_t operator[](int i) const
    {
        assert(i>=0 && i<length_);
        return streams[i];
    }

    inline ~CudaStreamArray()
    {
        free();
    }
};

/*
class CudaStreamNode
{
public:
    CudaStreamNode *next;
    CudaStreamNode *prev;
    CudaStream stream;
    
    CudaStreamNode()
        :next(0),prev(0)
    {
                    check_cuda_error(cudaStreamCreate(&stream),
                             "cudaCreateStreamCreate");
    }

    ~CudaStreamNode()
    {
                check_cuda_error(cudaStreamDestroy(stream),
                                 "cudaStreamDestroy");                
    }    
};


class CudaStreamPool
{
private:
    CudaStreamNode* free_streams;
    CudaStreamNode* allocated_streams;

public:    
    CudaStreamPool()
        :free_streams(0), allocated_streams(0)
    {}

    CudaStreamNode* allocate()
    {
        CudaStreamNode* node = 0;
        if (free_streams!=0) // get first node of free_streams
        {
            node = free_streams;

            // remove first node from free_streams
            free_streams = node->next;
            if (free_streams!=0)
                free_streams->prev = 0;
        }
        else // create new node 
        {
            node = new CudaStreamNode();                        
        }
        
        // insert node as first node of allocated_streams
        node->prev = 0;
        node->next = allocated_streams;
        if (allocated_streams!=0)
            allocated_streams->prev = node;
        allocated_streams = node;

        return node;
    }

    void free(CudaStreamNode* node)
    {
        // remove node from the list it's in
        if(node->prev!=0)
            node->prev->next = node->next;
        if(node->next!=0)
            node->next->prev = node->prev;

        // insert it as first node of free_streams
        node->prev = 0;
        node->next = free_streams;
        if (free_streams!=0)
            free_streams->prev = node;
        free_streams = node;
    }

    ~CudaStreamPool()
    {
    }    
}

class CudaEvent
{
private:
    static CudaEventPool event_pool;
    CudaEventNode* node;

public:

    CudaEvent()
    {
        node = event_pool.allocate();
    }

    ~CudaEvent()
    {
        event_pool.free(node);
    }
    
};
    

class CudaStream
{
private:
    static CudaStreamPool stream_pool;

    CudaStreamNode* node;

public:

    typedef CudaEvent event_t;
    
    CudaStream()
    {
        node = stream_pool.allocate();
    }

    void select()
    {
        check_status( cublasSetStream(handle(), node->stream) );
    }

    //! Records an event in the current stream. Scheduled to fire when the stream reaches it
    void record(const event_t& event)
    {
        check_cuda_error(cudaEventRecord(event.node->event, node->stream), "cudaEventRecord");
    }

    // Schedules a wating for that specific event in the current stream
    void wait(const event_t& event)
    {
        check_cuda_error(cudaStreamWaitEvent(node->stream, event.node->event, 0), "cudaStreamWaitEvent");
    }

    void synchronize()
    {
        cudaStreamSynchronize(node->stream);  
    }
    
    ~CudaStream()
    {
        stream_pool.free(node);
    }
};



class CublasPContext
{
    typedef CudaStream stream_t;
    typedef CudaEvent event_t;    
};

*/




class CudaEventArray
{
protected:
    mutable cudaEvent_t *events;
    mutable int length_;

public:

    inline CudaEventArray()
        :events(0), length_(0)
    {}

    inline CudaEventArray(int n)
         :events(new cudaEvent_t[n]),
         length_(n)
    {
        for (int i=0; i<length_; i++)
            check_cuda_error(cudaEventCreateWithFlags(&events[i], cudaEventDisableTiming),
                             "cudaCreateEvent");
    }

    inline void free()
    {
        if (events!=0)
        {
            for (int i=0; i<length_; i++)
                check_cuda_error(cudaEventDestroy(events[i]), "cudaEventDestroy");
            events = 0;
            length_ = 0;
        }
    }

    inline void operator=(const CudaEventArray& other)
    {
        events = other.events;
        length_ = other.length_;
        other.events = 0;  // steal the other's events array
        other.length_ = 0;
    }

    inline CudaEventArray(const CudaEventArray& other)
        :events(0), length_(0)
    { 
        operator=(other);
    }


    inline int length() const
    { return length_; }

    
    inline cudaEvent_t operator[](int i) const
    {
        assert(i>=0 && i<length_);
        return events[i];
    }

    inline ~CudaEventArray()
    {
        free();
    }
};

// static CudaStreamArray streams;

class CublasInit
{
private:
    cublasHandle_t handle_;
    
public:
    
    CudaStreamArray streams;
    
    inline cublasHandle_t handle() const
    {
        return handle_;
    }
    
    //! utility function to check the return status of a cublas call
    inline static void check_status(cublasStatus_t cublas_stat,  const string& doingwhat="") 
    { 
        if (cublas_stat != CUBLAS_STATUS_SUCCESS) 
            PLERROR("CUBLAS operation failed: " << doingwhat);
    }

    cudaStream_t createStream() const
    {
        cudaStream_t streamId;
        cudaError_t err = cudaStreamCreate (&streamId);
        if(err != cudaSuccess) 
            PLERROR("cudaCreateStream failed");
        return streamId;
    }

    void destroyStream(cudaStream_t streamId)
    {
        cudaError_t err = cudaStreamDestroy(streamId);
        if(err != cudaSuccess) 
            PLERROR("cudaStreamDestroy failed");
    }

    inline void setStream(cudaStream_t streamId) const
    { check_status( cublasSetStream(handle(), streamId) ); }

    inline cudaStream_t getStream() const
    { 
        cudaStream_t streamId;
        check_status( cublasGetStream(handle(), &streamId) ); 
        return streamId;
    }

    //! Records an event in the current stream. Scheduled to fire when the stream reaches it
    void recordEvent(cudaEvent_t event)
    {
        check_cuda_error(cudaEventRecord(event, getStream()), "cudaEventRecord");
    }

    // Schedules a wating for that specific event in the current stream
    void waitEvent(cudaEvent_t event)
    {
        check_cuda_error(cudaStreamWaitEvent(getStream(), event, 0), "cudaStreamWaitEvent");
    }

    inline void initialize()
    {
        int device_count = 0;
        check_cuda_error( cudaGetDeviceCount(&device_count) , "cudaGetDeviceCount"); 
        printf("Found %d cuda devices\n", device_count);

// numpy/arrayobject.h is included first if we are being included inside a Theano op
// if we detect this, then we do not attempt to initialize the GPU, as Theano already has

#ifndef Py_ARRAYOBJECT_H
        int card_nb = 0;
        const char* cuda_device_num = getenv("CUDA_DEVICE_NUM");
        if (cuda_device_num != 0)
            sscanf(cuda_device_num, "%d", &card_nb);

        
        check_cuda_error( cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync), "cudaSetDeviceFlags");

        printf("Doing cudaSetDevice(%d) as defined in environment variable CUDA_DEVICE_NUM"
               " (using 0 if undefined, set this environment variable properly if you want another device)", card_nb);
        check_cuda_error(cudaSetDevice(card_nb), " in cudaSetDevice");
#endif

        check_status( cublasCreate(&handle_), "Initializing cublas with cublasCreate");

        cublasPointerMode_t mode;
        check_status( cublasGetPointerMode(handle_, &mode), "getting intial pointer mode" );
    }

    inline void setPointerModeDevice() const
    { cublasSetPointerMode(handle(), CUBLAS_POINTER_MODE_DEVICE); }

    inline void setPointerModeHost() const
    { cublasSetPointerMode(handle(), CUBLAS_POINTER_MODE_HOST); }


    // used cuda device num will be looked up in environment variable CUDA_DEVICE_NUM
    inline CublasInit()
    {
        initialize();
        setPointerModeHost();
        streams = CudaStreamArray(5);
        setStream(streams[0]);
    }

    inline ~CublasInit()
    {
#ifndef Py_ARRAYOBJECT_H
        // only free these streams if we are not inside a Theano op 
        // (otherwise there's an error when doin so, possibly because Theano already 
        // freed cuda before the destructor of this static variable is called)
        streams.free();
#endif
        cublasDestroy(handle());
    }
};

static CublasInit cublas;


// Simple cublas overloads for float and double data types

template<class T>
inline cublasStatus_t cublas_copy(cublasHandle_t handle, int n, const T* x, int incx, T* y, int incy)
{
    cudaStream_t current_stream = cublas.getStream();
    cudaError_t err;
    if(incx==1 && incy==1) // special case of contiguous memory
    {
        err = cudaMemcpyAsync(y, x, n, cudaMemcpyDeviceToDevice, current_stream);
    }
    else // general case with increments
    {
        err = cudaMemcpy2DAsync(y, incy*sizeof(T), x, incx*sizeof(T), sizeof(T), n, cudaMemcpyDeviceToDevice, current_stream);
    }
    
    if (err == cudaSuccess)
        return CUBLAS_STATUS_SUCCESS;
    else
        return CUBLAS_STATUS_INTERNAL_ERROR;
}

inline cublasStatus_t cublas_copy(cublasHandle_t handle, int n, const float *x, int incx, float *y, int incy) 
{  return cublasScopy(handle, n, x, incx, y, incy); }

inline cublasStatus_t cublas_copy(cublasHandle_t handle, int n, const double *x, int incx, double *y, int incy) 
{  return cublasDcopy(handle, n, x, incx, y, incy); }

inline void cuda_memclear(int* ptr, int n)
{ cudaMemset(ptr, 0, n*sizeof(int)); }

inline void cuda_memclear(float* ptr, int n)
{ cudaMemset(ptr, 0, n*sizeof(float)); }

inline void cuda_memclear(double* ptr, int n)
{ cudaMemset(ptr, 0, n*sizeof(double)); }

// clears memory of mxn column-major matrix with stride lda 
template<class T>
inline void cuda_memclear(T* ptr, int nrows, int ncols, int stride)
{
    T* p = ptr;
    for(int j=0; j<ncols; j++, p+=stride)
        cuda_memclear(p, nrows);
}

// specialization for flat and double can call special case of cublas_geam
template<>
inline void cuda_memclear(float* ptr, int nrows, int ncols, int stride)
{ 
    float zero = 0;
    cublas.setPointerModeHost();
    cublas.check_status( cublasSgeam(cublas.handle(), CUBLAS_OP_N, CUBLAS_OP_N, nrows, ncols, &zero, ptr, stride, &zero, ptr, stride, ptr, stride) );
}

template<>
inline void cuda_memclear(double* ptr, int nrows, int ncols, int stride)
{ 
    double zero = 0;
    cublas.setPointerModeHost();
    cublas.check_status( cublasDgeam(cublas.handle(), CUBLAS_OP_N, CUBLAS_OP_N, nrows, ncols, &zero, ptr, stride, &zero, ptr, stride, ptr, stride) );
}




// Note: semantic of const for CublasVec and CublasMat classes: 
// A vector or matrix is considered to remain const as long as its *structure* does not change
// The data content of a const& CublasVec or CublasMat can be changed.
// This allows ex to pass A.column(3) directly as the argument of a function taking in a const CublasVec& 
// without the compiler complaining, and avoids extra complications.
// So in summary a const CublasVec or CublasMat can have its data changed, but not its structure (dimensions, ...)

// predeclarations
template<class T> class CublasMat;

// *************************
// *   CublasVec class
// *************************
// This CublasVec class is meant to be but a thin wrapper around a cublas vector residing in device memory

template<class T> 
class CublasVec {

    friend class CublasMat<T>;

protected:
    T* data_;     // pointer to first element 
    int length_;  // number of elements
    int step_;    // step to next element of vector 
    PP< CudaStorage<T> > storage_;  // smartpointer to allocated memory
    
public:
    typedef T elem_t;    
    
    inline T* data() const
    { return data_; }

    inline const T* const_data() const
    { return data_; }

    inline int length() const
    { return length_; }
    
    inline int step() const
    { return step_; }

    //! Default constructor
    inline CublasVec()
        :data_(0), length_(0), step_(0), storage_(0)
    {}


  //! constructor from existing already allocated memory chunk. Not managed by storage.
    inline CublasVec(T* data, int length, int step, CudaStorage<T>* storage=0)
      :data_(data), length_(length), step_(step), storage_(storage)
  {}

  //! copy constructor. Copies the structure, not the data.
  inline CublasVec(const CublasVec<T>& x)
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
    inline explicit CublasVec(int length)
    :length_(length), step_(1)
  {
      storage_ = new CudaStorage<T> (length_);  // Will be automatically deleted when its refcount becomes zero
      assert( storage_->refcount() == 1);
      data_ = storage_->dataptr_;
  }

  //! operator= Copies the structure, not the data 
  // (use operator << if you want to copy the data between two vectors)
  inline const CublasVec<T>& operator=(const CublasVec<T>& x)
  {
    data_ = x.data_;
    length_ = x.length_;
    step_ = x.step_;
    storage_ = x.storage_;
    return *this;
  }
  
  //! y << x   (copies elements of x to this vector y) vectors must not overlap
  inline void operator<< (const CublasVec<T>& x) const
  {
    assert(this->length_==x.length_);
    cublas.check_status( cublas_copy(cublas.handle(), length_, x.const_data(), x.step_, this->data_, this->step_) );
  }

  //! x >> y   (copies elements of this vector x to y) vectors must not overlap
  inline void operator>> (const CublasVec<T>& y) const 
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
        
        if (data_==0)  // we've never allocated this vec yet
        {
            (*this) = CublasVec(new_length);
        }
        else if( new_length<=length_ )
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

                CublasVec<T> newv(new_length);
                if (copy_data)
                    newv.subVec(0,length_) << (*this); 
                (*this) = newv;
            }
        }
        else
            PLERROR("Resizing vector to larger size not supported if there's no storage");
    }
    
  //! fills the vector with the specified element
    void fill(const T& val) const;

  //! fills with 0
  inline void clear() const
  { 
      if (is_contiguous())
          cuda_memclear(data(), length());
      else
          cuda_memclear(data(), 1, length(), step());
  }


  //! read-only access to element at position i (0-based indexing)
  inline T get(int i) const
  {
      assert(i>=0 && i<length_);
      T val;
      cudaError_t stat = cudaMemcpy(&val, data_+i*step_, sizeof(T), cudaMemcpyDeviceToHost);
      if (stat!=cudaSuccess)
          PLERROR("cudaMemcpy call failed");
      return val;
  }

  inline void set(int i, const T& val) const
    {
      assert(i>=0 && i<length_);
      cudaError_t stat = cudaMemcpyAsync(data_+i*step_, &val, sizeof(T), cudaMemcpyHostToDevice, cublas.getStream());
      if (stat!=cudaSuccess)
          PLERROR("cudaMemcpy call failed");
    }

  //! read-only access to element at position i (0-based indexing)
  inline T operator[](int i) const
  { return get(i); }

  //! read-only access element at position i (0-based indexing)
  inline T operator()(int i) const
  { return get(i); }
  
  //! returns a sub-vector
  inline CublasVec<T> subVec(int startpos, int subvec_length) const
  {
    assert(subvec_length>=0 && startpos+subvec_length <=length_);
    CublasVec<T> subv(data_+startpos*step_, subvec_length, step_, storage_);
    return subv;
  }


    //! Returns a vector (allocated on the device) filled with 0 of the specified length
    //! Do NEVER modify the elements of this vector, as other calls will rely on it being filled with zeros
    static CublasVec<T> zeros(int d)
    {
        static CublasVec<T> device_zeros;
        if ( d>device_zeros.length() )
        {
            int n = max(d*2, 1024);
            device_zeros = CublasVec<T>(n);
            device_zeros.clear();
        }
        return device_zeros.subVec(0,d);                
    }

    //! Returns a vector (allocated on the device) filled with 1 of the specified length
    //! Do NEVER modify the elements of this vector, as other calls will rely on it being filled with ones
    static CublasVec<T> ones(int d)
    {
        static CublasVec<T> device_ones;
        if ( d>device_ones.length() )
        {
            int n = max(d*2, 1024);
            device_ones = CublasVec<T>(n);
            device_ones << BVec<T>::ones(n);
        }
        return device_ones.subVec(0,d);                
    }


};



float* get_cuda_cst_float()
{
    // Note TODO: these should probably rather be put in constant memory.
    // It's just that as far as I know constant memory alloc need to be compiled with nvcc
    // and declare as e.g. __constant__ float cst_ptr [size];

    static CublasVec<float> cst_float;
    if (cst_float.isEmpty())
    {
        const int n = 3;
        float v[n] = {0.0f, 1.0f, 2.0f};
        cst_float.resize(n);
        cublas.check_status( cublasSetVector(3, sizeof(float), v, 1, cst_float.data(), cst_float.step()) ); 
    }

    return cst_float.data();
}

#define CUDA_FLOAT_ZERO_PTR (get_cuda_cst_float())
#define CUDA_FLOAT_ONE_PTR (get_cuda_cst_float()+1)
#define CUDA_FLOAT_TWO_PTR (get_cuda_cst_float()+2)


double* get_cuda_cst_double()
{
    // Note TODO: these should probably rather be put in constant memory.
    // It's just that as far as I know constant memory alloc need to be compiled with nvcc
    // and declare as e.g. __constant__ double cst_ptr [size];

    static CublasVec<double> cst_double;
    if (cst_double.isEmpty())
    {
        const int n = 3;
        double v[n] = {0.0, 1.0, 2.0};
        cst_double.resize(n);
        cublas.check_status( cublasSetVector(3, sizeof(double), v, 1, cst_double.data(), cst_double.step()) ); 
    }

    return cst_double.data();
}

#define CUDA_DOUBLE_ZERO_PTR (get_cuda_cst_double())
#define CUDA_DOUBLE_ONE_PTR (get_cuda_cst_double()+1)
#define CUDA_DOUBLE_TWO_PTR (get_cuda_cst_double()+2)


// *************************
// *   CublasMat class
// *************************
// This CublasMat class is meant to be but a thin wrapper around a column-major cublas matrix residing in device memory

template<class T>
class CublasMat {

protected:
    T* data_; /* pointer to first element of matrix in column-major storage */
    int nrows_;
    int ncols_;
    int stride_; /* step to next column (setp to next row is always 1 in blas column major matrices) */
    PP< CudaStorage<T> > storage_;  // smartpointer to allocated memory

public:
    typedef T elem_t;    
    typedef CublasVec<T> vec_t;        
    
    inline T* data() const { return data_; }
    inline const T* const_data() const { return data_; }
    inline int nrows() const { return nrows_; }
    inline int ncols() const { return ncols_; }
    inline int stride() const { return stride_; }
    
    inline int storage_nrows() const { return storage_->nrows_; }

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
  inline CublasMat()
      :data_(0), nrows_(0), ncols_(0), stride_(0), storage_(0)
  {}
  
  //! constructor from existing already allocated memory chunk (if you do not want this to be automatically memory managed, pass 0 for storage)
    inline CublasMat(T* data, int nrows, int ncols, int stride, CudaStorage<T>* storage=0)
      :data_(data), nrows_(nrows), ncols_(ncols), stride_(stride), storage_(storage)
  {
      assert(data!=0 || (nrows==0 && ncols==0) );      
  }

  //! copy constructor. Copies the structure, not the data. Results in another view on the same data.
  inline CublasMat(const CublasMat<T>& m)
      :data_(m.data_), nrows_(m.nrows_), ncols_(m.ncols_), stride_(m.stride_),
       storage_(m.storage_)
    {}

    //! View of a contiguous vector as a single column matrix or single row matrix
    inline CublasMat(const CublasVec<T>& x, bool as_row=false)
        :data_(x.data()), nrows_(x.length()), ncols_(1), storage_(x.storage_)
    {
      if (!as_row)  // view it a sa single column matrix, must be contiguous
          assert( x.is_contiguous() );
      else // view it a sa single row matrix
      {
          nrows_ = 1;
          ncols_ = x.length();
          stride_ = x.step_;
      }      
      
    }

  //! Constructs a matrix view of a contiguous vector
  inline CublasMat(const CublasVec<T>& x, int nrows, int ncols)
      :data_(x.data()), nrows_(nrows_), ncols_(ncols), storage_(x.storage_)
  {
    assert( x.is_contiguous() );
    assert( nrows*ncols==x.length() );
  }

  //! Returns a flat vector view of a contiguous matrix
  inline CublasVec<T> flat() const
  {
    assert( is_contiguous() );
    return CublasVec<T>(data_, nrows_*ncols_, 1, storage_);
  }

    //! Returns a flat vector view of the memory range of this matrix
    //!
    inline CublasVec<T> flat_including_pitch_margin() const
    {
        if (is_contiguous()) // return flat view of contiguous data
            return CublasVec<T>(data_, nrows_*ncols_, 1, storage_);
        else if (nrows_ == storage_nrows()) // return flat view of data area that includes a margin due to pitch  
            return CublasVec<T>(data_, stride_*ncols_, 1, storage_);
        else
        {
            PLERROR("flat_including_pitch_margin invoked on a matrix that is neither contiguous nor has its stride equal to the storage allocated stride/pitch");
            return CublasVec<T>(); // to make compiler happy
        }
    }


    //! constructor that allocates memory
    inline CublasMat(int nrows, int ncols)
        :nrows_(nrows), ncols_(ncols)
    {
        assert(nrows>=1 && ncols>=1);
        
        storage_ = new CudaStorage<T> (nrows, ncols);  // Will be automatically deleted when its refcount becomes zero
        assert( storage_->refcount() == 1);

        stride_ = storage_->stride_;
        data_ = storage_->dataptr_;        
    }


  //! operator= Copies the structure, not the data, i.e. makes anothe view on the same data
  // (use operator << if you want to copy the data between two vectors)
  inline const CublasMat<T>& operator=(const CublasMat<T>& m)
  {
    data_ = m.data_;
    nrows_ = m.nrows_;
    ncols_ = m.ncols_;
    stride_ = m.stride_;
    storage_ = m.storage_;
    return *this;
  }

    //! C <- A (copies elements) matrices must not overlap
    inline void operator<< (const CublasMat<T>& A) const
    {
        assert( A.nrows_==nrows_ && A.ncols_==ncols_ );
        assert( data_!=0 );
       
        // cudaMemcpy2D has row-major semantics, this is why we use nrows_ as the "width" and ncols_ as the "height"
        cudaMemcpy2DAsync(data_, stride_*sizeof(T), A.const_data(), A.stride()*sizeof(T), nrows_*sizeof(T), ncols_, cudaMemcpyDeviceToDevice, cublas.getStream());

        // printf("  checking operator<< went well on %d x %d matrix : max_abs_diff = %e \n",nrows_, ncols_, max_abs_diff(A, *this));

           // OLD IMPLEMENTATION 
/*
        // Detect contiguous memory, if so we can copy in one shot
        if ( A.is_contiguous() && this->is_contiguous() )
            this->flat() << A.flat();
        else if ( this->nrows_==this->storage_nrows() && this->nrows_==A.storage_nrows() 
                  && this->storage_->stride_ == A.storage_->stride_)  // Contiguous if it wasn't for the pitch allocation margin
            this->flat_including_pitch_margin() << A.flat_including_pitch_margin();
        else // Non contiguous, need to copy column by column
        {
            for (int j=0; j<this->ncols_; j++)
                this->column(j) << A.column(j);
        }
*/
        // printf("  checking operator<< went well on %d x %d matrix : max_abs_diff = %e \n",nrows_, ncols_, max_abs_diff(A, *this));

    }

  //! A >> C  (copies elements of this matrix A to matrix C) matrices must not overlap
  inline void operator>> (const CublasMat<T>& C) const
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
            (*this) = CublasMat<T>(new_nrows, new_ncols);
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

                CublasMat<T> newm(new_nrows, new_ncols);
                if (copy_data)
                    newm.subMatColumns(0,ncols_) << (*this);
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
            this->flat().fill(val);
        else if (nrows_==storage_nrows())  // contiguous if it wasn't for the pitch allocation margin (that we can overwrite)
            this->flat_including_pitch_margin().fill(val);
        else // need to fill column by column
        {
            for(int j=0; j<ncols_; j++)
                this->column(j).fill(val);
        }
    }

    //! fills the matrix with zeros
    inline void clear() const
    { 
        if (is_contiguous())
            cuda_memclear(data_, nrows_*ncols_);
        else if (nrows_ == storage_nrows())  // contiguous except for pitch alloc margin
            cuda_memclear(data_, stride_*ncols_);
        else  // actually call geam to clear the matrix data
            cuda_memclear(data_, nrows_, ncols_, stride_);
    }
    
    //! read-only access to element at position i (0-based index)
    inline T get(int i, int j) const
    {
      assert(i>=0 && i<nrows_ && j>=0 && j<ncols_);
      T val;
      cudaError_t stat = cudaMemcpy(&val, data_+i+stride_*j, sizeof(T), cudaMemcpyDeviceToHost);
      if (stat!=cudaSuccess)
          PLERROR("cudaMemcpy call failed");
      return val;
    }

    inline void set(int i, int j, const T& val) const
    {
      assert(i>=0 && i<nrows_ && j>=0 && j<ncols_);
      cudaError_t stat = cudaMemcpy(data_+i+stride_*j, &val, sizeof(T), cudaMemcpyHostToDevice);
      if (stat!=cudaSuccess)
          PLERROR("cudaMemcpy call failed");
    }

    //! read-only access to element at position i (0-based index)
    inline T operator() (int i, int j) const
    { return get(i,j); }
    
    //! returns a vector that is a row of this matrix
    // index argument is 0 based (i.e. between 0 and nrows-1 inclusive)
    inline CublasVec<T> row(int i) const
    {
        assert(i>=0 && i<nrows_);
        return CublasVec<T>(data_+i, ncols_, stride_, storage_);
    }
  
    //! returns a vector that is a column of this matrix
    // index argument is 0 based (i.e. between 0 and ncols-1 inclusive)
    inline CublasVec<T> column(int j) const
    {
        assert(j>=0 && j<ncols_);
        return CublasVec<T>(data_+j*stride_, nrows_, 1, storage_);
    }
  
    //! Returns a CublasVec view of the main diagonal of this matrix 
    inline CublasVec<T> diagonal() const
    {
        CublasVec<T> di( data_, min(nrows_, ncols_), stride_+1, storage_ );
        return di;
    }

    //! Returns a CublasVec view of the main diagonal of this matrix (alias for diagonal() )
    inline CublasVec<T> diag() const
    { return diagonal(); }

    //! returns a sub-matrix
    // index arguments i,j are 0 based 
    inline CublasMat<T> subMat(int i, int j, int nrows, int ncols) const
    {
      assert(i>=0 && j>=0 && nrows>=1 && ncols>=1 && i+nrows<=nrows_ && j+ncols<=ncols_);
      CublasMat<T> subm(data_+i+j*stride_, nrows, ncols, stride_, storage_);
      return subm;
    }

    //! returns a sub-matrix corresponding to a range of full columns
    // index arguments j is 0 based 
    inline CublasMat<T> subMatColumns(int j, int ncols) const
    {
      assert(j>=0 && ncols>=1 && j+ncols<=ncols_);
      CublasMat<T> subm(data_+j*stride_, nrows_, ncols, stride_, storage_);
      return subm;
    }

    //! returns a sub-matrix corresponding to a range of full rows
    // index arguments i is 0 based 
    inline CublasMat<T> subMatRows(int i, int nrows) const
    {
      assert(i>=0 && nrows>=1 && i+nrows<=nrows_);
      CublasMat<T> subm(data_+i, nrows, ncols_, stride_, storage_);
      return subm;
    }
  
  };


  /* ******************************************************************** */
  /*   Copies of vectors and matrices between host and cuda device        */
  /*   i.e. between BVec<T> and CublasVec<T> and BMat<T> and CublasMat<T> */
  /* ******************************************************************** */

// Vector transfer operations

template<class T>
inline cublasStatus_t cublasSetVector(const BVec<T>& src, const CublasVec<T>& dest)
{ return ::cublasSetVector(src.length(), sizeof(T), src.data(), src.step(), dest.data(), dest.step()); }

template<class T>
inline cublasStatus_t cublasSetVectorAsync(const BVec<T>& src, const CublasVec<T>& dest, cudaStream_t stream)
{ return ::cublasSetVectorAsync(src.length(), sizeof(T), src.data(), src.step(), dest.data(), dest.step(), stream); }

template<class T>
inline cublasStatus_t cublasGetVector(const CublasVec<T>& src, const BVec<T>& dest)
{ return ::cublasGetVector(src.length(), sizeof(T), src.data(), src.step(), dest.data(), dest.step()); }

template<class T>
inline cublasStatus_t cublasGetVectorAsync(const CublasVec<T>& src, const BVec<T>& dest, cudaStream_t stream)
{ return ::cublasGetVectorAsync(src.length(), sizeof(T), src.data(), src.step(), dest.data(), dest.step(), stream); }


template<class T>
inline void operator>>(const BVec<T>& src, const CublasVec<T>& dest)
{
    assert(dest.length() == src.length());
    cublas.check_status( cublasSetVectorAsync(src,dest,cublas.getStream()) );
}

template<class T>
inline void operator>>(const CublasVec<T>& src, const BVec<T>& dest)
{
    assert(dest.length() == src.length());
    cudaStream_t current_stream = cublas.getStream();    
    cublas.check_status( cublasGetVectorAsync(src,dest,current_stream) );
    cudaStreamSynchronize(current_stream); // wait for memory transfer to finish
}

template<class T>
inline void operator<<(const CublasVec<T>& dest, const BVec<T>& src)
{ src >> dest; }

template<class T>
inline void operator<<(const BVec<T>& dest, const CublasVec<T>& src)
{ src >> dest; }

// Matrix transfer operations

template<class T>
inline cublasStatus_t cublasSetMatrix(const BMat<T>& src, const CublasMat<T>& dest)
{ return ::cublasSetMatrix(src.nrows(), src.ncols(), sizeof(T), 
                           src.data(), src.stride(), dest.data(), dest.stride()); }


template<class T>
inline cublasStatus_t cublasSetMatrixAsync(const BMat<T>& src, const CublasMat<T>& dest, cudaStream_t stream)
{ return ::cublasSetMatrixAsync(src.nrows(), src.ncols(), sizeof(T), 
                                src.data(), src.stride(), dest.data(), dest.stride(), stream); }


template<class T>
inline cublasStatus_t cublasGetMatrix(const CublasMat<T>& src, const BMat<T>& dest)
{ return ::cublasGetMatrix(src.nrows(), src.ncols(), sizeof(T), 
                         src.data(), src.stride(), dest.data(), dest.stride()); }

template<class T>
inline cublasStatus_t cublasGetMatrixAsync(const CublasMat<T>& src, const BMat<T>& dest, cudaStream_t stream)
{ return ::cublasGetMatrixAsync(src.nrows(), src.ncols(), sizeof(T), 
                                src.data(), src.stride(), dest.data(), dest.stride(), stream); }


template<class T>
inline void operator>>(const BMat<T>& src, const CublasMat<T>& dest)
{
    assert(dest.nrows()==src.nrows() && dest.ncols()==src.ncols() );
    cublas.check_status( cublasSetMatrixAsync(src,dest,cublas.getStream()) );
}

template<class T>
inline void operator>>(const CublasMat<T>& src, const BMat<T>& dest)
{
    assert(dest.nrows() == src.nrows() && dest.ncols() == src.ncols() );
    cudaStream_t current_stream = cublas.getStream();
    cublas.check_status( cublasGetMatrixAsync(src,dest,current_stream) );
    cudaStreamSynchronize(current_stream); // wait for memory transfer to finish
}

template<class T>
inline void operator<<(const CublasMat<T>& dest, const BMat<T>& src)
{ src >> dest; }

template<class T>
inline void operator<<(const BMat<T>& dest, const CublasMat<T>& src)
{ src >> dest; }




/* ******************************************************* */
/*   Define generic functions for CublasVec<T> and CublasMat<T>        */
/* ******************************************************* */


// -------------------------
// Basic I/O

// Todo: these ops currently transfer elements one by one which is probably very slow
// It might be better to transfer whole vectorsand matrices at once.

template<class T> 
void print(const CublasVec<T>& x)
{
    printf("CublasVec (n=%d, step=%d): [  ", x.length(), x.step());
    for (int i=0; i<x.length(); i++)
    {
        print(x[i]);
        printf(" ");
    }
    printf("   ]  \n");
}

template<class T> 
void print(const CublasMat<T>& A)
{
    printf("CublasMat %d x %d  (stride=%d)\n", A.nrows(), A.ncols(), A.stride());
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


// TO DO TODO

  //! Fills vector with integers uniformly sampled between low and high inclusive 
// template<class T>
//void fill_random_uniform_discrete(const CublasVec<T>& x, int low, int high)

  //! Fills vector with integers uniformly sampled between low and high inclusive, keeps resmpling if we already sampled the value
  //template<class T>
  //void fill_random_uniform_discrete_noduplicate(const CublasVec<T>& x, int low, int high)

  //! Fills matrix with integers uniformly sampled between low and high inclusive 
  //template<class T>
  //void fill_random_uniform_discrete(const CublasMat<T>& m, int low, int high)

  //! Fills vector with real numbers (converted to T) uniformly sampled between low and high 
  //template<class T>
  //void fill_random_uniform_continuous(const CublasVec<T>& x, double low, double high)

  //! Fills matrix A with real numbers (converted to T) uniformly sampled between low and high 
  //template<class T>
  //void fill_random_uniform_continuous(const CublasMat<T>& A, double low, double high)


  // ------------------------------------
  // Blas Linear algebra operations 


// vector data swap
inline void blas_swap(const CublasVec<float>& x, const CublasVec<float>& y)
{
    assert( x.length() == y.length() );
    cublas.check_status( cublasSswap( cublas.handle(), x.length(), 
                                     x.data(), x.step(), y.data(), y.step()) );   
}

//! computes x' y  

inline float blas_dot(const CublasVec<float>& x, const CublasVec<float>& y)
{ 
    float result = 0;
    cublas.setPointerModeHost();
    cublas.check_status( cublasSdot( cublas.handle(), x.length(), 
                                     x.data(), x.step(), y.data(), y.step(), &result) );
    return result;
}

inline void blas_dot_ondevice(const CublasVec<float>& x, const CublasVec<float>& y, float* result_on_device)
{ 
    assert(x.length()==y.length()); 
    cublas.setPointerModeDevice();
    cublas.check_status( cublasSdot( cublas.handle(), x.length(), 
                                     x.data(), x.step(), y.data(), y.step(), result_on_device) );
}

inline double blas_dot(const CublasVec<double>& x, const CublasVec<double>& y)
{ 
    double result = 0;
    cublas.setPointerModeHost();
    cublas.check_status( cublasDdot( cublas.handle(), x.length(), 
                                     x.data(), x.step(), y.data(), y.step(), &result) );
    return result;
}

inline void blas_dot_ondevice(const CublasVec<double>& x, const CublasVec<double>& y, double* result_on_device)
{ 
    assert(x.length()==y.length()); 
    cublas.setPointerModeDevice();
    cublas.check_status( cublasDdot( cublas.handle(), x.length(), 
                                     x.data(), x.step(), y.data(), y.step(), result_on_device) );
}

//! x = alpha x   
inline void blas_scal(float alpha, const CublasVec<float>& x)
{
    cublas.setPointerModeHost();
    cublas.check_status( cublasSscal( cublas.handle(), x.length(), &alpha, x.data(), x.step() ) ); 
}

inline void blas_scal(double alpha, const CublasVec<double>& x)
{
    cublas.setPointerModeHost();
    cublas.check_status( cublasDscal( cublas.handle(), x.length(), &alpha, x.data(), x.step() ) ); 
}

inline void blas_scal_ondevice(float* alpha_on_device, const CublasVec<float>& x)
{
    cublas.setPointerModeDevice();
    cublas.check_status( cublasSscal( cublas.handle(), x.length(), alpha_on_device, x.data(), x.step() ) ); 
}

inline void blas_scal_ondevice(double* alpha_on_device, const CublasVec<double>& x)
{
    cublas.setPointerModeDevice();
    cublas.check_status( cublasDscal( cublas.handle(), x.length(), alpha_on_device, x.data(), x.step() ) ); 
}

inline void operator*=(const CublasVec<float>& x, float alpha)
{ blas_scal( alpha, x); }

inline void operator*=(const CublasVec<double>& x, double alpha)
{ blas_scal( alpha, x); }

//! A = alpha A
inline void blas_scal(float alpha, const CublasMat<float>& A)
{ 
    if (A.is_contiguous())          
        blas_scal(alpha, A.flat() ); 
    else if (A.nrows() == A.storage_nrows() ) // contiguous except for pitch alloc margin
        blas_scal(alpha, A.flat_including_pitch_margin() );
    else
    {  // TODO: could should be made faster
        for (int j=0; j<A.ncols(); j++)
            blas_scal(alpha, A.column(j));
    }
}

//! A = alpha A
inline void blas_scal(double alpha, const CublasMat<double>& A)
{ 
    if (A.is_contiguous())          
        blas_scal(alpha, A.flat() ); 
    else if (A.nrows() == A.storage_nrows() ) // contiguous except for pitch alloc margin
        blas_scal(alpha, A.flat_including_pitch_margin() );
    else
    {  // TODO: could should be made faster
        for (int j=0; j<A.ncols(); j++)
            blas_scal(alpha, A.column(j));
    }
}

// C = alpha op_A(A) + beta op_B(B)
// where op_A and op_B can be 'N' for noop or 'T' for transpose
// see documentaiton of cublas geam form more info
inline void blas_geam(float alpha, const CublasMat<float>& A, char op_A, 
                      float beta, const CublasMat<float>& B, char op_B, 
                      const CublasMat<float>& C)
{
    int m = C.nrows();
    int n = C.ncols();
    assert(    (op_A=='N' && A.nrows()==m && A.ncols()==n)
               || (op_A=='T' && A.nrows()==n && A.ncols()==m) );
    assert(    (op_B=='N' && B.nrows()==m && B.ncols()==n)
               || (op_B=='T' && B.nrows()==n && B.ncols()==m) );

    cublas.setPointerModeHost();
    cublas.check_status( cublasSgeam( cublas.handle(), toCublasOp(op_A), toCublasOp(op_B), m, n, 
                                      &alpha, A.data(), A.stride(),
                                      &beta, B.data(), B.stride(),
                                      C.data(), C.stride() ) );
}

// C = alpha op_A(A) + beta op_B(B)
// whre op_A and op_B can be 'N' for noop or 'T' for transpose
// see documentaiton of cublas geam form more info
inline void blas_geam(double alpha, const CublasMat<double>& A, char op_A, 
                      double beta, const CublasMat<double>& B, char op_B, 
                      const CublasMat<double>& C)
{
    int m = C.nrows();
    int n = C.ncols();
    assert(    (op_A=='N' && A.nrows()==m && A.ncols()==n)
               || (op_A=='T' && A.nrows()==n && A.ncols()==m) );
    assert(    (op_B=='N' && B.nrows()==m && B.ncols()==n)
               || (op_B=='T' && B.nrows()==n && B.ncols()==m) );

    cublas.setPointerModeHost();
    cublas.check_status( cublasDgeam( cublas.handle(), toCublasOp(op_A), toCublasOp(op_B), m, n, 
                                      &alpha, A.data(), A.stride(),
                                      &beta, B.data(), B.stride(),
                                      C.data(), C.stride() ) );
}



inline void operator*=(const CublasMat<float>& A, float alpha)
{ blas_scal( alpha, A); }

inline void operator*=(const CublasMat<double>& A, double alpha)
{ blas_scal( alpha, A); }

inline void operator/=(const CublasMat<float>& A, float alpha)
{ A *= (1.0f/alpha); }

inline void operator/=(const CublasMat<float>& A, double alpha)
{ A *= (1.0/alpha); }

//! y = y + alpha x 
inline void blas_axpy(float alpha, const CublasVec<float>& x, const CublasVec<float>& y)
{ 
    assert(x.length()==y.length()); 
    cublas.setPointerModeHost();
    cublas.check_status( cublasSaxpy(cublas.handle(), x.length(), &alpha, x.data(), x.step(), y.data(), y.step()) );
}

//! y = y + alpha x 
inline void blas_axpy(double alpha, const CublasVec<double>& x, const CublasVec<double>& y)
{ 
    assert(x.length()==y.length()); 
    cublas.setPointerModeHost();
    cublas.check_status( cublasDaxpy(cublas.handle(), x.length(), &alpha, x.data(), x.step(), y.data(), y.step()) );
}

//! x += y
inline void operator+=(const CublasVec<float>& x, const CublasVec<float>& y)
{ blas_axpy( 1.0f, y, x); } 

//! x += y
inline void operator+=(const CublasVec<double>& x, const CublasVec<double>& y)
{ blas_axpy( 1.0, y, x); } 

//! x -= y
inline void operator-=(const CublasVec<float>& x, const CublasVec<float>& y)
{ blas_axpy( -1.0f, y, x); } 

//! x -= y
inline void operator-=(const CublasVec<double>& x, const CublasVec<double>& y)
{ blas_axpy( -1.0, y, x); } 

//! x += alpha    add scalar to vector
inline void operator+=(const CublasVec<float>& x, float alpha)
{
    int n = x.length();
    CublasVec<float> ones = CublasVec<float>::ones(n);
    cublas.setPointerModeHost();
    cublas.check_status( cublasSaxpy(cublas.handle(), n, &alpha, ones.data(), ones.step(), x.data(), x.step()) );
}
    
//! x += alpha    add scalar to vector
inline void operator+=(const CublasVec<double>& x, double alpha)
{
    int n = x.length();
    CublasVec<double> ones = CublasVec<double>::ones(n);
    cublas.setPointerModeHost();
    cublas.check_status( cublasDaxpy(cublas.handle(), n, &alpha, ones.data(), ones.step(), x.data(), x.step()) );
}

//! Y = Y + alpha X 
inline void blas_axpy(float alpha, const CublasMat<float>& X, const CublasMat<float>& Y)
{ 
    blas_geam(alpha, X, 'N', 1, Y, 'N', Y);
}

//! Y = Y + alpha X 
inline void blas_axpy(double alpha, const CublasMat<double>& X, const CublasMat<double>& Y)
{ 
    blas_geam(alpha, X, 'N', 1, Y, 'N', Y);
}

//! X += Y
inline void operator+=(const CublasMat<float>& X, const CublasMat<float>& Y)
{ blas_axpy( 1.0f, Y, X); } 

//! X += Y
inline void operator+=(const CublasMat<double>& X, const CublasMat<double>& Y)
{ blas_axpy( 1.0, Y, X); } 

//! X -= Y
inline void operator-=(const CublasMat<float>& X, const CublasMat<float>& Y)
{ blas_axpy( -1.0f, Y, X); } 

//! X -= Y
inline void operator-=(const CublasMat<double>& X, const CublasMat<double>& Y)
{ blas_axpy( -1.0, Y, X); } 


//! A = A + alpha x y'  
inline void blas_ger(float alpha, const CublasVec<float>& x, const CublasVec<float>& y, const CublasMat<float>& A)
{ 
    assert(A.nrows()==x.length() && A.ncols()==y.length());
    cublas.setPointerModeHost();
    cublas.check_status( cublasSger(cublas.handle(), x.length(), y.length(), &alpha, x.data(), x.step(), y.data(), y.step(), A.data(), A.stride()) );
}

//! A = A + alpha x y'  
inline void blas_ger(double alpha, const CublasVec<double>& x, const CublasVec<double>& y, const CublasMat<double>& A)
{ 
    assert(A.nrows()==x.length() && A.ncols()==y.length());
    cublas.setPointerModeHost();
    cublas.check_status( cublasDger(cublas.handle(), x.length(), y.length(), &alpha, x.data(), x.step(), y.data(), y.step(), A.data(), A.stride()) );
}

//! A = A + alpha x x'    
//! A is supposed symmetric and we use only its lower part (if uplo='L') or upper part (if uplo='U') 
inline void blas_syr(char uplo, float alpha, const CublasVec<float>& x, const CublasMat<float>& A)
{ 
    assert( uplo=='L' || uplo=='U' );
    assert(A.nrows()==x.length() && A.ncols()==x.length()); 
    cublas.setPointerModeHost();
    cublas.check_status( cublasSsyr(cublas.handle(), toCublasFillMode(uplo), x.length(), &alpha, 
                                    x.data(), x.step(), A.data(), A.stride()) ); 
}

//! A = A + alpha x x'    
//! A is supposed symmetric and we use only its lower part (if uplo='L') or upper part (if uplo='U') 
inline void blas_syr(char uplo, double alpha, const CublasVec<double>& x, const CublasMat<double>& A)
{ 
    assert( uplo=='L' || uplo=='U' );
    assert(A.nrows()==x.length() && A.ncols()==x.length()); 
    cublas.setPointerModeHost();
    cublas.check_status( cublasDsyr(cublas.handle(), toCublasFillMode(uplo), x.length(), &alpha, 
                                    x.data(), x.step(), A.data(), A.stride()) ); 
}

//! A = A + alpha x y' + alpha y x'   
//! A is supposed symmetric, considering only its lower part (if uplo='L') or upper part (if uplo='U') 
inline void blas_syr2(char uplo, float alpha, const CublasVec<float>& x, const CublasVec<float>& y, const CublasMat<float>& A)
{ 
    assert( uplo=='L' || uplo=='U' );
    assert(x.length()==y.length() && A.nrows()==x.length() && A.ncols()==A.nrows()); 
    cublas.setPointerModeHost();
    cublas.check_status( cublasSsyr2(cublas.handle(), toCublasFillMode(uplo), x.length(), &alpha, 
                                     x.data(), x.step(), y.data(), y.step(), A.data(), A.stride()) ); 
}

//! A = A + alpha x y' + alpha y x'   
//! A is supposed symmetric, considering only its lower part (if uplo='L') or upper part (if uplo='U') 
inline void blas_syr2(char uplo, double alpha, const CublasVec<double>& x, const CublasVec<double>& y, const CublasMat<double>& A)
{ 
    assert( uplo=='L' || uplo=='U' );
    assert(x.length()==y.length() && A.nrows()==x.length() && A.ncols()==A.nrows()); 
    cublas.setPointerModeHost();
    cublas.check_status( cublasDsyr2(cublas.handle(), toCublasFillMode(uplo), x.length(), &alpha, 
                                     x.data(), x.step(), y.data(), y.step(), A.data(), A.stride()) ); 
}

//! y = alpha A x + beta y   (if trans=='N')
//! y = alpha A^T x + beta y   (if trans=='T')
inline void blas_gemv(char trans, float alpha, const CublasMat<float>& A, const CublasVec<float>& x, float beta, const CublasVec<float>& y)
{ 
    assert( (trans=='N' && A.nrows()==y.length() && A.ncols()==x.length())  || 
            (trans=='T' && A.nrows()==x.length() && A.ncols()==y.length()) );
    cublas.setPointerModeHost();
    cublas.check_status( cublasSgemv(cublas.handle(), toCublasOp(trans), A.nrows(), A.ncols(), 
                                     &alpha, A.data(), A.stride(), x.data(), x.step(), 
                                     &beta, y.data(), y.step() ) ); 
}

//! y = alpha A x + beta y   (if trans=='N')
//! y = alpha A^T x + beta y   (if trans=='T')
inline void blas_gemv(char trans, double alpha, const CublasMat<double>& A, const CublasVec<double>& x, double beta, const CublasVec<double>& y)
{ 
    assert( (trans=='N' && A.nrows()==y.length() && A.ncols()==x.length())  || 
            (trans=='T' && A.nrows()==x.length() && A.ncols()==y.length()) );
    cublas.setPointerModeHost();
    cublas.check_status( cublasDgemv(cublas.handle(), toCublasOp(trans), A.nrows(), A.ncols(), 
                                     &alpha, A.data(), A.stride(), x.data(), x.step(), 
                                     &beta, y.data(), y.step() ) ); 
}



//! y = alpha A x + beta y  
//! A is a symmetric matrix, and we use only its lower part (if uplo='L') or upper part (if uplo='U') 
inline void blas_symv(char uplo, float alpha, const CublasMat<float>& A, const CublasVec<float>& x, float beta, const CublasVec<float>& y)
{ 
    assert( uplo=='L' || uplo=='U' );
    assert(A.nrows()==A.ncols() && A.ncols()==x.length() && A.nrows()==y.length()); 
    cublas.setPointerModeHost();
    cublas.check_status( cublasSsymv(cublas.handle(), toCublasFillMode(uplo), y.length(), 
                                     &alpha, A.data(), A.stride(), x.data(), x.step(), 
                                     &beta, y.data(), y.step() ) ); 
}

//! y = alpha A x + beta y  
//! A is a symmetric matrix, and we use only its lower part (if uplo='L') or upper part (if uplo='U') 
inline void blas_symv(char uplo, double alpha, const CublasMat<double>& A, const CublasVec<double>& x, double beta, const CublasVec<double>& y)
{ 
    assert( uplo=='L' || uplo=='U' );
    assert(A.nrows()==A.ncols() && A.ncols()==x.length() && A.nrows()==y.length()); 
    cublas.setPointerModeHost();
    cublas.check_status( cublasDsymv(cublas.handle(), toCublasFillMode(uplo), y.length(), 
                                     &alpha, A.data(), A.stride(), x.data(), x.step(), 
                                     &beta, y.data(), y.step() ) ); 
}

//! C = alpha A B + beta C 
inline void blas_gemm_NN(float alpha, const CublasMat<float>& A, const CublasMat<float>& B, float beta, const CublasMat<float>& C)
{  
    assert(A.nrows()==C.nrows() && B.ncols()==C.ncols() && A.ncols()==B.nrows());
    cublas.setPointerModeHost();
    cublas.check_status( cublasSgemm(cublas.handle(), CUBLAS_OP_N, CUBLAS_OP_N, C.nrows(), C.ncols(), A.ncols(),            
                                     &alpha, A.data(), A.stride(), B.data(), B.stride(), 
                                     &beta, C.data(), C.stride()) ); 
}

//! C = alpha A B + beta C 
inline void blas_gemm_NN(double alpha, const CublasMat<double>& A, const CublasMat<double>& B, double beta, const CublasMat<double>& C)
{  
    assert(A.nrows()==C.nrows() && B.ncols()==C.ncols() && A.ncols()==B.nrows());
    cublas.setPointerModeHost();
    cublas.check_status( cublasDgemm(cublas.handle(), CUBLAS_OP_N, CUBLAS_OP_N, C.nrows(), C.ncols(), A.ncols(),            
                                     &alpha, A.data(), A.stride(), B.data(), B.stride(), 
                                     &beta, C.data(), C.stride()) ); 
}


//! C = alpha A' B + beta C 
inline void blas_gemm_TN(float alpha, const CublasMat<float>& A, const CublasMat<float>& B, float beta, const CublasMat<float>& C)
{  
    assert(A.ncols()==C.nrows() && B.ncols()==C.ncols() && A.nrows()==B.nrows()); 
    cublas.setPointerModeHost();
    cublas.check_status( cublasSgemm(cublas.handle(), CUBLAS_OP_T, CUBLAS_OP_N, C.nrows(), C.ncols(), A.nrows(), 
                                     &alpha, A.data(), A.stride(), B.data(), B.stride(), 
                                     &beta, C.data(), C.stride()) ); 
}

//! C = alpha A' B + beta C 
inline void blas_gemm_TN(double alpha, const CublasMat<double>& A, const CublasMat<double>& B, double beta, const CublasMat<double>& C)
{  
    assert(A.ncols()==C.nrows() && B.ncols()==C.ncols() && A.nrows()==B.nrows()); 
    cublas.setPointerModeHost();
    cublas.check_status( cublasDgemm(cublas.handle(), CUBLAS_OP_T, CUBLAS_OP_N, C.nrows(), C.ncols(), A.nrows(), 
                                     &alpha, A.data(), A.stride(), B.data(), B.stride(), 
                                     &beta, C.data(), C.stride()) ); 
}


//! C = alpha A B' + beta C 
inline void blas_gemm_NT(float alpha, const CublasMat<float>& A, const CublasMat<float>& B, float beta, const CublasMat<float>& C)
{  
    assert(A.nrows()==C.nrows() && B.nrows()==C.ncols() && A.ncols()==B.ncols());
    cublas.setPointerModeHost();
    cublas.check_status( cublasSgemm(cublas.handle(), CUBLAS_OP_N, CUBLAS_OP_T, C.nrows(), C.ncols(), A.ncols(), 
                                     &alpha, A.data(), A.stride(), B.data(), B.stride(), 
                                     &beta, C.data(), C.stride()) ); 
}

//! C = alpha A B' + beta C 
inline void blas_gemm_NT(double alpha, const CublasMat<double>& A, const CublasMat<double>& B, double beta, const CublasMat<double>& C)
{  
    assert(A.nrows()==C.nrows() && B.nrows()==C.ncols() && A.ncols()==B.ncols());
    cublas.setPointerModeHost();
    cublas.check_status( cublasDgemm(cublas.handle(), CUBLAS_OP_N, CUBLAS_OP_T, C.nrows(), C.ncols(), A.ncols(), 
                                     &alpha, A.data(), A.stride(), B.data(), B.stride(), 
                                     &beta, C.data(), C.stride()) ); 
}


//!  C = alpha A' B' + beta C 
inline void blas_gemm_TT(float alpha, const CublasMat<float>& A, const CublasMat<float>& B, float beta, const CublasMat<float>& C)
{  
    assert(A.ncols()==C.nrows() && B.nrows()==C.ncols() && A.nrows()==B.ncols());
    cublas.setPointerModeHost();
    cublas.check_status( cublasSgemm(cublas.handle(), CUBLAS_OP_T, CUBLAS_OP_T, C.nrows(), C.ncols(), A.nrows(), 
                                     &alpha, A.data(), A.stride(), B.data(), B.stride(), 
                                     &beta, C.data(), C.stride()) ); 
}

//!  C = alpha A' B' + beta C 
inline void blas_gemm_TT(double alpha, const CublasMat<double>& A, const CublasMat<double>& B, double beta, const CublasMat<double>& C)
{  
    assert(A.ncols()==C.nrows() && B.nrows()==C.ncols() && A.nrows()==B.ncols());
    cublas.setPointerModeHost();
    cublas.check_status( cublasDgemm(cublas.handle(), CUBLAS_OP_T, CUBLAS_OP_T, C.nrows(), C.ncols(), A.nrows(), 
                                     &alpha, A.data(), A.stride(), B.data(), B.stride(), 
                                     &beta, C.data(), C.stride()) ); 
}






//! C = alpha A B + beta C 
//! A is a symmetric matrix, and we use only its lower part (if uplo='L') or upper part (if uplo='U')
inline void blas_symm(char uplo, float alpha, const CublasMat<float>& A, const CublasMat<float>& B, float beta, const CublasMat<float>& C)
{ 
    assert( uplo=='L' || uplo=='U' );
    assert(A.nrows()==A.ncols() && A.ncols()==B.nrows() && B.nrows()==C.nrows() && B.ncols()==C.ncols());
    cublas.setPointerModeHost();
    cublasSideMode_t leftside = CUBLAS_SIDE_LEFT;    
    cublas.check_status( cublasSsymm(cublas.handle(), leftside, toCublasFillMode(uplo), C.nrows(), C.ncols(), 
                                     &alpha, A.data(), A.stride(), B.data(), B.stride(), 
                                     &beta, C.data(), C.stride()) ); 
}

  //! C = alpha B A + beta C 
  //! A is a symmetric matrix, and we use only its lower part (if uplo='L') or upper part (if uplo='U')
inline void blas_symm_rightside(char uplo, float alpha, const CublasMat<float>& A, const CublasMat<float>& B, float beta, const CublasMat<float>& C)
{ 
    assert( uplo=='L' || uplo=='U' );
    assert(A.nrows()==A.ncols() && A.nrows()==B.ncols() && B.nrows()==C.nrows() && B.ncols()==C.ncols());
    cublas.setPointerModeHost();
    cublasSideMode_t rightside = CUBLAS_SIDE_RIGHT;    
    cublas.check_status( cublasSsymm(cublas.handle(), rightside, toCublasFillMode(uplo),
                                     C.nrows(), C.ncols(), 
                                     &alpha, A.data(), A.stride(), B.data(), B.stride(), 
                                     &beta, C.data(), C.stride()) ); 
}

//! C = alpha A B + beta C 
//! A is a symmetric matrix, and we use only its lower part (if uplo='L') or upper part (if uplo='U')
inline void blas_symm(char uplo, double alpha, const CublasMat<double>& A, const CublasMat<double>& B, double beta, const CublasMat<double>& C)
{ 
    assert( uplo=='L' || uplo=='U' );
    assert(A.nrows()==A.ncols() && A.ncols()==B.nrows() && B.nrows()==C.nrows() && B.ncols()==C.ncols());
    cublas.setPointerModeHost();
    cublasSideMode_t leftside = CUBLAS_SIDE_LEFT;    
    cublas.check_status( cublasDsymm(cublas.handle(), leftside, toCublasFillMode(uplo), C.nrows(), C.ncols(), 
                                     &alpha, A.data(), A.stride(), B.data(), B.stride(), 
                                     &beta, C.data(), C.stride()) ); 
}

  //! C = alpha B A + beta C 
  //! A is a symmetric matrix, and we use only its lower part (if uplo='L') or upper part (if uplo='U')
inline void blas_symm_rightside(char uplo, double alpha, const CublasMat<double>& A, const CublasMat<double>& B, double beta, const CublasMat<double>& C)
{ 
    assert( uplo=='L' || uplo=='U' );
    assert(A.nrows()==A.ncols() && A.nrows()==B.ncols() && B.nrows()==C.nrows() && B.ncols()==C.ncols());
    cublas.setPointerModeHost();
    cublasSideMode_t rightside = CUBLAS_SIDE_RIGHT;    
    cublas.check_status( cublasDsymm(cublas.handle(), rightside, toCublasFillMode(uplo),
                                     C.nrows(), C.ncols(), 
                                     &alpha, A.data(), A.stride(), B.data(), B.stride(), 
                                     &beta, C.data(), C.stride()) ); 
}

//! C = alpha A A' + beta C 
//! C is a symmetric matrix, and we use only its lower part (if uplo='L') or upper part (if uplo='U')
inline void blas_syrk(char uplo, float alpha, const CublasMat<float>& A, float beta, const CublasMat<float>& C)
{ 
    assert( uplo=='L' || uplo=='U' );
    assert(C.nrows()==C.ncols() && A.nrows()==C.nrows()); 
    cublas.setPointerModeHost();
    cublas.check_status( cublasSsyrk(cublas.handle(), toCublasFillMode(uplo), CUBLAS_OP_N, A.nrows(), A.ncols(), 
                                     &alpha, A.data(), A.stride(), 
                                     &beta, C.data(), C.stride() ) ); 
}

//! C = alpha A A' + beta C 
//! C is a symmetric matrix, and we use only its lower part (if uplo='L') or upper part (if uplo='U')
inline void blas_syrk(char uplo, double alpha, const CublasMat<double>& A, double beta, const CublasMat<double>& C)
{ 
    assert( uplo=='L' || uplo=='U' );
    assert(C.nrows()==C.ncols() && A.nrows()==C.nrows()); 
    cublas.setPointerModeHost();
    cublas.check_status( cublasDsyrk(cublas.handle(), toCublasFillMode(uplo), CUBLAS_OP_N, A.nrows(), A.ncols(), 
                                     &alpha, A.data(), A.stride(), 
                                     &beta, C.data(), C.stride() ) ); 
}

//! C = alpha A' A + beta C 
//! C is a symmetric matrix, and we consider only its lower part (if uplo='L') or upper part (if uplo='U') */
inline void blas_syrk_T(char uplo, float alpha, const CublasMat<float>& A, float beta, const CublasMat<float>& C)
{ 
    assert( uplo=='L' || uplo=='U' );
    assert(C.nrows()==C.ncols() && A.ncols()==C.nrows()); 
    cublas.setPointerModeHost();
    cublas.check_status( cublasSsyrk(cublas.handle(), toCublasFillMode(uplo), CUBLAS_OP_T, A.ncols(), A.nrows(), 
                                     &alpha, A.data(), A.stride(), 
                                     &beta, C.data(), C.stride() ) ); 
}

//! C = alpha A' A + beta C 
//! C is a symmetric matrix, and we consider only its lower part (if uplo='L') or upper part (if uplo='U') */
inline void blas_syrk_T(char uplo, double alpha, const CublasMat<double>& A, double beta, const CublasMat<double>& C)
{ 
    assert( uplo=='L' || uplo=='U' );
    assert(C.nrows()==C.ncols() && A.ncols()==C.nrows()); 
    cublas.setPointerModeHost();
    cublas.check_status( cublasDsyrk(cublas.handle(), toCublasFillMode(uplo), CUBLAS_OP_T, A.ncols(), A.nrows(), 
                                     &alpha, A.data(), A.stride(), 
                                     &beta, C.data(), C.stride() ) ); 
}

//! C = alpha A B' + B A' + beta C 
//! C is a symmetric matrix, and we use only its lower part (if uplo='L') or upper part (if uplo='U') 
inline void blas_syr2k(char uplo, float alpha, const CublasMat<float>& A, const CublasMat<float>& B, float beta, const CublasMat<float>& C)
{ 
    assert( uplo=='L' || uplo=='U' );    
    assert(C.nrows()==C.ncols() && A.nrows()==C.nrows() && B.nrows()==C.nrows() && A.ncols()==B.ncols());
    cublas.setPointerModeHost();
    cublas.check_status( cublasSsyr2k(cublas.handle(), toCublasFillMode(uplo), CUBLAS_OP_N, A.nrows(), A.ncols(), 
                                      &alpha, A.data(), A.stride(), B.data(), B.stride(), 
                                      &beta, C.data(), C.stride() ) ); 
}

//! C = alpha A B' + B A' + beta C 
//! C is a symmetric matrix, and we use only its lower part (if uplo='L') or upper part (if uplo='U') 
inline void blas_syr2k(char uplo, double alpha, const CublasMat<double>& A, const CublasMat<double>& B, double beta, const CublasMat<double>& C)
{ 
    assert( uplo=='L' || uplo=='U' );    
    assert(C.nrows()==C.ncols() && A.nrows()==C.nrows() && B.nrows()==C.nrows() && A.ncols()==B.ncols());
    cublas.setPointerModeHost();
    cublas.check_status( cublasDsyr2k(cublas.handle(), toCublasFillMode(uplo), CUBLAS_OP_N, A.nrows(), A.ncols(), 
                                      &alpha, A.data(), A.stride(), B.data(), B.stride(), 
                                      &beta, C.data(), C.stride() ) ); 
}

//! C = alpha A' B + B' A + beta C 
//! C is a symmetric matrix, and we use only its lower part (if uplo='L') or upper part (if uplo='U')
inline void blas_syr2k_T(char uplo, float alpha, const CublasMat<float>& A, const CublasMat<float>& B, float beta, const CublasMat<float>& C)
{ 
    assert( uplo=='L' || uplo=='U' );    
    int n = C.nrows();
    assert(C.ncols()==n && A.ncols()==n && B.ncols()==n && A.nrows()==B.nrows()); 
    cublas.setPointerModeHost();
    cublas.check_status( cublasSsyr2k(cublas.handle(), toCublasFillMode(uplo), CUBLAS_OP_T, A.ncols(), A.nrows(), 
                                      &alpha, A.data(), A.stride(), B.data(), B.stride(), 
                                      &beta, C.data(), C.stride() ) ); 
}

//! C = alpha A' B + B' A + beta C 
//! C is a symmetric matrix, and we use only its lower part (if uplo='L') or upper part (if uplo='U')
inline void blas_syr2k_T(char uplo, double alpha, const CublasMat<double>& A, const CublasMat<double>& B, double beta, const CublasMat<double>& C)
{ 
    assert( uplo=='L' || uplo=='U' );    
    int n = C.nrows();
    assert(C.ncols()==n && A.ncols()==n && B.ncols()==n && A.nrows()==B.nrows()); 
    cublas.setPointerModeHost();
    cublas.check_status( cublasDsyr2k(cublas.handle(), toCublasFillMode(uplo), CUBLAS_OP_T, A.ncols(), A.nrows(), 
                                      &alpha, A.data(), A.stride(), B.data(), B.stride(), 
                                      &beta, C.data(), C.stride() ) ); 
}

// Cublas- blas-like extension op

void cublas_dgmm(cublasSideMode_t mode, const CublasMat<float>& A, const CublasVec<float>& x, const CublasMat<float>& C)
{
    int m = A.nrows();
    int n = A.ncols();
    assert(C.nrows()==m && C.ncols()==n);
    assert( (mode==CUBLAS_SIDE_LEFT && x.length()==m) || 
            (mode==CUBLAS_SIDE_RIGHT && x.length()==n) );

    cublas.setPointerModeHost();
    cublas.check_status( cublasSdgmm(cublas.handle(), mode, m, n, 
                                     A.data(), A.stride(), x.data(), x.step(), 
                                     C.data(), C.stride()) );    
}

void cublas_dgmm(cublasSideMode_t mode, const CublasMat<double>& A, const CublasVec<double>& x, const CublasMat<double>& C)
{
    int m = A.nrows();
    int n = A.ncols();
    assert(C.nrows()==m && C.ncols()==n);
    assert( (mode==CUBLAS_SIDE_LEFT && x.length()==m) || 
            (mode==CUBLAS_SIDE_RIGHT && x.length()==n) );

    cublas.setPointerModeHost();
    cublas.check_status( cublasDdgmm(cublas.handle(), mode, m, n, 
                                     A.data(), A.stride(), x.data(), x.step(), 
                                     C.data(), C.stride()) );    
}



//!  A = A diag(x)  
inline void scale_columns(const CublasMat<float>& A, const CublasVec<float>& scales) 
{  cublas_dgmm(CUBLAS_SIDE_RIGHT, A, scales, A);  }

//!  B = A diag(x)  
inline void scale_columns(const CublasMat<float>& A, const CublasVec<float>& scales, const CublasMat<float>& B) 
{  cublas_dgmm(CUBLAS_SIDE_RIGHT, A, scales, B);  }

//!  A = A diag(x)  
inline void scale_columns(const CublasMat<double>& A, const CublasVec<double>& scales) 
{  cublas_dgmm(CUBLAS_SIDE_RIGHT, A, scales, A);  }

//!  B = A diag(x)  
inline void scale_columns(const CublasMat<double>& A, const CublasVec<double>& scales, const CublasMat<double>& B) 
{  cublas_dgmm(CUBLAS_SIDE_RIGHT, A, scales, A);  }

//!  A = diag(x) A  
inline void scale_rows(const CublasMat<float>& A, const CublasVec<float>& scales) 
{  cublas_dgmm(CUBLAS_SIDE_LEFT, A, scales, A);  }

//!  B = diag(x) A  
inline void scale_rows(const CublasMat<float>& A, const CublasVec<float>& scales, const CublasMat<float>& B) 
{  cublas_dgmm(CUBLAS_SIDE_LEFT, A, scales, B);  }

//!  A = diag(x) A  
inline void scale_rows(const CublasMat<double>& A, const CublasVec<double>& scales) 
{  cublas_dgmm(CUBLAS_SIDE_LEFT, A, scales, A);  }

//!  B = diag(x) A  
inline void scale_rows(const CublasMat<double>& A, const CublasVec<double>& scales, const CublasMat<double>& B) 
{  cublas_dgmm(CUBLAS_SIDE_LEFT, A, scales, B);  }

//! x *= y
template<class T>
inline void operator*=(const CublasVec<T>& x, const CublasVec<T>& y)
{
    CublasMat<T> X(x, true);
    scale_columns(X,y);
}


/* OLD VERSION
//!  A = A diag(x)  
inline void scale_columns(const CublasMat<T>& A, const CublasVec<T>& scales) 
  { 
      
      int n = A.nrows();
      assert(scales.length() == A.ncols());
      T* Ak = A.data();
      for(int k=0; k<A.ncols(); k++, Ak += A.stride())       
          blas_scal(n, scales[k], Ak, 1); 
  } 

  //!  A = diag(x) A  
  template<class T>
  inline void scale_rows(const CublasMat<T>& A, const CublasVec<T>& scales) 
  { 
      assert(scales.length() == A.nrows());    
    T* Aj = A.data();
    for(int j=0; j<A.ncols(); j++, Aj += A.stride())
        for(int i=0; i<A.nrows(); i++)
            Aj[i] *= scales[i];
  } 
*/

// ------------------------------------
// Generic Linear algebra operations


//! diag(A) <- diag(A) + alpha
inline void add_scalar_to_diagonal(float alpha, const CublasMat<float>& A) 
{  A.diagonal() += alpha; }

//! diag(A) <- diag(A) + alpha
inline void add_scalar_to_diagonal(double alpha, const CublasMat<double>& A) 
{  A.diagonal() += alpha; }


template<class T>
void transpose(const CublasMat<T>& src, const CublasMat<T>& dest)
{
    blas_geam((T)1, src, 'T', (T)0, dest, 'N', dest);
}

template<class T>
void transpose_squaremat_inplace(const CublasMat<T>& A)
{
    assert(A.nrows()==A.ncols()); 
   
    int n = A.nrows();
    for(int i=0; i<n-1; i++)
        blas_swap( A.column(i).subVec(i+1, n-(i+1)), 
                   A.row(i).subVec(i+1, n-(i+1)) );
}

//! Copies the lower part of a square matrix (below the diagonal) to the upper part. Thus making it a symmetric matrix.
template<class T>
void copy_lower_to_upper(const CublasMat<T>& A)
{
    assert(A.nrows()==A.ncols());
    
    int n = A.nrows();
    for(int i=0; i<n-1; i++)
        A.column(i).subVec(i+1, n-(i+1)) >> A.row(i).subVec(i+1, n-(i+1));
}

//! Copies the upper part of a square matrix (above the diagonal) to the lower part. Thus making it a symmetric matrix.
template<class T>
void copy_upper_to_lower(const CublasMat<T>& A)
{
    assert(A.nrows()==A.ncols());

    int n = A.nrows();
    for(int i=0; i<n-1; i++)
        A.column(i).subVec(i+1, n-(i+1)) << A.row(i).subVec(i+1, n-(i+1));
}


//! for each row of the matrix, computes the sum of its elements
template<class T> 
inline void sum_rowwise(const CublasMat<T>& A, const CublasVec<T>& x)
{
    CublasVec<T> ones = CublasVec<T>::ones(A.ncols());
    blas_gemv('N', T(1), A, ones, (T)0, x);
}


//! for each column of the matrix, computes the sum of its elements
template<class T> 
inline void sum_columnwise(const CublasMat<T>& A, const CublasVec<T>& x)
{
    CublasVec<T> ones = CublasVec<T>::ones(A.nrows());
    blas_gemv('T', T(1), A, ones, (T)0, x);
}



//! Returns sum( A * B )  where * denotes elementwise product and sum is the sum over all elements.
template<class T>
inline T sum_prod(const CublasMat<T>& A, const CublasMat<T>& B)
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



//! returns the sum of the elements of vector x
//! (currently done by calling blas_dot with vector of ones)
template<class T> 
inline T sum(const CublasVec<T>& x)
{
    return blas_dot(CublasVec<T>::ones(x.length()), x);
    /*
    // Old version: copy on host and perform sum there
    static BVec<T> host_x;
    host_x.resize(x.length());
    host_x << x;
    return sum(host_x);
    */
}

//! X += y (y considered a column vector)
  template<class T>
  inline void addColumnVector(const CublasMat<T>& X, const CublasVec<T>& y)
  {
      blas_ger((T)1, y, CublasVec<T>::ones(X.ncols()), X);
  }

//! X += y^T (y^T considered a row vector)
  template<class T>
  inline void addRowVector(const CublasMat<T>& X, const CublasVec<T>& y)
  {
      blas_ger((T)1, CublasVec<T>::ones(X.nrows()), y, X);
  }


template<class T>
inline T trace(const CublasMat<T>& A)
{
    assert(A.nrows()==A.ncols()); // trace is usually for squared matrices
    return sum(A.diagonal());
}

//! returns the product of the elements of vector x
// CURRENTLY INEFFICIENT: done on host
template<class T> 
inline T product(const CublasVec<T>& x)
{
    // for now copy on host and perform sum there
    BVec<T> host_x(x.length());
    host_x << x;
    return product(host_x);
}


//! inverts each element of x i.e. x[i] = 1/x[i]
// CURRENTLY INEFFICIENT: done on host
template<class T> 
inline void invert_elements(const CublasVec<T>& x)
{
    // for now copy on host and invert there
    BVec<T> host_x(x.length());
    host_x << x;
    invert_elements(host_x);
    // copy result back to device
    host_x >> x;
}

//! takes the square root of each element of x i.e. x[i] = sqrt(x[i])
// CURRENTLY INEFFICIENT: done on host
template<class T> 
inline void sqrt_elements(const CublasVec<T>& x)
{
    // for now copy on host and perform operation there
    BVec<T> host_x(x.length());
    host_x << x;
    sqrt_elements(host_x);
    // copy result back to device
    host_x >> x;
}

//! z = x-y
template<class T> 
inline void diff(const CublasVec<T>& x, const CublasVec<T>& y, const CublasVec<T>& z)
{
    z << x;
    z -= y;
}

//! C = A-B
template<class T> 
inline void diff(const CublasMat<T>& A, const CublasMat<T>& B, const CublasMat<T>& C)
{
    blas_geam(T(1), A, 'N', T(-1), B, 'N', C); 
}


// TODO: Following implementations are currently extremely inefficient
// CPU does all the computation and elements are accessed (copied from the device) one by one 

  template<class T>
  T min(const CublasMat<T>& A)
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
  T max(const CublasMat<T>& A)
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
  T max_abs(const CublasMat<T>& A)
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
  T max_abs_diff(const CublasVec<T>& x, const CublasVec<T>& y)
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
  T max_abs_diff(const CublasMat<T>& A, const CublasMat<T>& B)
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


  template<class T>
  T max_abs_diff(const CublasMat<T>& A, const BMat<T>& B)
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

template<class T>
T max_abs_diff(const BMat<T>& A, const CublasMat<T>& B)
{
    return max_abs_diff(B,A);
}

//! returns the index (0-based) of the element with the largest absolute value
template<class T> 
inline int argmax_abs(const CublasVec<T>& x)
{
    static BVec<T> x_cpu;
    x_cpu.resize(x.length());
    x_cpu << x;
    return argmax_abs(x_cpu);
}


//////////////////////////////////////////////////////
//////////////////////////////////////////////////////
//////////////////////////////////////////////////////
//////////////////////////////////////////////////////
//////////////////////////////////////////////////////


//! C = C+AB  where all matrices use dense representation, but B is mostly zeros (sparse).
//! The same result can be obtained by calling gemm, but here we avoid unnecessary
//! computation due to the many elements of B that are 0.
template<class T>
void accumulate_product_with_densely_stored_sparse_mat(const CublasMat<T>& A, const CublasMat<T>& B, const CublasMat<T>& C)
{
    assert(A.nrows()==C.nrows() && B.ncols()==C.ncols() && A.ncols()==B.nrows());
    for (int j=0; j<B.ncols(); j++)
    {
        CublasVec<T> Bj = B.column(j);
        CublasVec<T> Cj = C.column(j);
        for(int i=0; i<B.nrows(); i++)
        {
            T Bij = Bj[i];
            if (Bij!=0)
                blas_axpy( Bij, A.column(i), Cj );
        }
    }
}

template<class T>
void product_with_densely_stored_sparse_mat(const CublasMat<T>& A, const CublasMat<T>& B, const CublasMat<T>& C)
{
    clear(C);
    accumulate_product_with_densely_stored_sparse_mat(A, B, C);
}


  //!  C = A diag(x)  
  template<class T>
  inline void product_matrix_diagonal(const CublasMat<T>& A, const CublasVec<T>& x, const CublasMat<T>& C) 
  {   
    C << A;
    scale_columns(C,x);
  } 



  //! fills the vector with the specified element
    // VERY INEFFICIENT. But there are slightly better specializations for float and double
// TODO: should really launch a elemwise kernel for filling!
template<typename T>
void CublasVec<T>::fill(const T& val) const
  {
      if (length()>0)
      {
          // Create tmpvec on host and copy it over
          BVec<T> tmpvec(length());
          tmpvec.fill(val);
          tmpvec >> *this;
      }
  }



// specialization for somewhat more efficient versions of fill 
// TODO: should really launch a elemwise kernel for filling!
template<>
void CublasVec<float>::fill(const float& val) const
{
    if (length()>0)
    {
        if (val==0)
            clear();
        else
        {
            (*this) << CublasVec<float>::ones(length());
            if (val!=1.0f)
                blas_scal( val, *this); 
        }
    }
}

// specialization for somewhat more efficient versions of fill 
// TODO: should really launch a elemwise kernel for filling!
template<> 
void CublasVec<double>::fill(const double& val) const
{
    if (length()>0)
    {
        if (val==0)
            clear();
        else
        {
            (*this) << CublasVec<double>::ones(length());
            if (val!=1.0)
                blas_scal( val, *this); 
        }
    }
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
T power_iteration(const CublasMat<T>& A, const CublasVec<T>& x, int niter=10)
{
    static CublasVec<T> other_x;

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
T power_iteration_for_singular_vect(const CublasMat<T>& A, const CublasVec<T>& left_singvec, const CublasVec<T>& right_singvec, int niter=100)
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
// Cublas Linear algebra operations

//! Perform LU factorization of a single square matrix A in place
//! pivots must be of length  
//! A will be modified to contain the superposition of L and U.
//! The info execution status is returned

int cublas_getrf(const CublasMat<float>& A, const CublasVec<int>& pivots)
{
    static CublasVec<int> infos(1); // will contain the info execution status int 
    static CublasVec<float*> Aarray(1); // will contain a pointer to the data of A

    if(A.nrows()!=A.ncols())
        PLERROR("A must be a square matrix");
    if(pivots.length()!=A.nrows())
        PLERROR("pivots must be allocated with same length as A");

    Aarray.set(0, A.data());  
    cublas.check_status( cublasSgetrfBatched(cublas.handle(),
                                             A.nrows(), 
                                             Aarray.data(),
                                             A.stride(), 
                                             pivots.data(),
                                             infos.data(),
                                             1) );    
    return infos[0];
}


//! This call will solve a system of equations of the form AX = B
//! A must be a (n,n) square matrix.
//! pivots must be passed and be of length n (will be filled by the call). 
//! The call destroys A and B, filling B with solution X, and A with the superposed pivoted LU factorization
void cublasSolveLinearSystem(const CublasMat<float>& A, const CublasMat<float>& B, const CublasVec<int>& pivots)
{
    static CublasVec<int> infos(1); // will contain the info execution status int 
    static CublasVec< float* > Aarray(1); // will contain a pointer to the data of A
    static CublasVec< float* > Barray(1); // will contain a pointer to the data of B

    int n = A.nrows();
    if(A.ncols()!=n)
        PLERROR("A must be a square matrix");
    if(B.nrows()!=n)
        PLERROR("A and B have incompatible sizes");
    if(pivots.length()!=n)
        PLERROR("pivots must be allocated with same length as A");

    // First compute LU factorization with pivoting
    // printf("First column of A BEFORE\n");
    // print(A.column(0));
    // printf("\n");
    
    Aarray.set(0, A.data()); 

    //CublasMat<float> A_copy(A.nrows(), A.ncols());
    //A_copy << A;
    //printf("Max Abs Difference before: %e \n", max_abs_diff(A,A_copy));

    cublas.check_status( cublasSgetrfBatched(cublas.handle(),
                                             A.nrows(), 
                                             Aarray.data(),
                                             A.stride(), 
                                             pivots.data(),
                                             infos.data(),
                                             1) );
    int getrf_info = infos[0];
    if (getrf_info > 0)
        PLERROR("U is exactly singular");
    if (getrf_info < 0) 
        PLERROR("Parameter #" << (-getrf_info) << " passed to cublas_getrfBatched has an illegal value");

    // printf("First column of A AFTER\n");
    // print(A.column(0));
    // printf("\n");

    // printf("Max Abs Difference after: %e \n", max_abs_diff(A,A_copy));

    // Second finish solving linear system
    Barray.set(0, B.data()); 
    
//    printf("Aarray[0] = %p \n", (void*)Aarray[0]);
    int getrs_info;
    
    cublas.check_status( cublasSgetrsBatched(cublas.handle(),
                                             CUBLAS_OP_N, 
                                             n, 
                                             B.ncols(), 
                                             const_cast<const float**>(Aarray.const_data()), 
                                             A.stride(), 
                                             pivots.const_data(),
                                             Barray.data(),
                                             B.stride(),
                                             &getrs_info,
                                             1) );

    if (getrs_info < 0) 
        PLERROR("Parameter #" << (-getrs_info) << " passed to cublas_getrsBatched has an illegal value");

    // printf("getrs_info = %d\n", getrs_info);
}



//! Does C <- A^-1
//! A and C must be (n,n) square matrix.
//! The call destroys A (which will contain the superposed pivoted LU factorization).
void cublasInvertMatrix_destructive(const CublasMat<float>& A, const CublasMat<float>& C)
{
    static CublasVec<int> pivots;
    static CublasVec<int> infos(1); // will contain the info execution status int 
    static CublasVec< float* > Aarray(1); // will contain a pointer to the data of A
    static CublasVec< float* > Carray(1); // will contain a pointer to the data of C

    int n = A.nrows();
    if(A.ncols()!=n)
        PLERROR("A must be a square matrix");
    if(C.nrows()!=n || C.ncols()!=n)
        PLERROR("A and C must have the same dimensions");

    pivots.resize(n,false);

    // First compute LU factorization with pivoting
    // printf("First column of A CEFORE\n");
    // print(A.column(0));
    // printf("\n");
    
    Aarray.set(0, A.data()); 

    //CublasMat<float> A_copy(A.nrows(), A.ncols());
    //A_copy << A;
    //printf("Max Abs Difference before: %e \n", max_abs_diff(A,A_copy));

    cublas.check_status( cublasSgetrfBatched(cublas.handle(),
                                             A.nrows(), 
                                             Aarray.data(),
                                             A.stride(), 
                                             pivots.data(),
                                             infos.data(),
                                             1) );
    int getrf_info = infos[0];
    if (getrf_info > 0)
        PLERROR("U is exactly singular");
    if (getrf_info < 0) 
        PLERROR("Parameter #" << (-getrf_info) << " passed to cublas_getrfBatched has an illegal value");

    // printf("First column of A AFTER\n");
    // print(A.column(0));
    // printf("\n");

    // printf("Max Abs Difference after: %e \n", max_abs_diff(A,A_copy));

    // Second finish solving linear system
    Carray.set(0, C.data()); 
    
//    printf("Aarray[0] = %p \n", (void*)Aarray[0]);

    cublas.check_status( cublasSgetriBatched(cublas.handle(),
                                             n, 
                                             const_cast<const float**>(Aarray.data()),
                                             A.stride(), 
                                             pivots.data(),
                                             Carray.data(),
                                             C.stride(),
                                             infos.data(),
                                             1) );

    int getri_info = infos[0];
    if (getri_info != 0)
        PLERROR("Inversion failed: is exactly singular: " << getri_info);
}

//! Does C <- A^-1
//! A and C must be (n,n) square matrix.
//! The call does not modify A
void cublasInvertMatrix(const CublasMat<float>& A, const CublasMat<float>& Ainv)
{
    static CublasMat<float> C;
    C.resize(A.nrows(), A.ncols(), false);
    A >> C;
    cublasInvertMatrix_destructive(C, Ainv);
}

//! Does A <- A^-1
//! A must be (n,n) square matrix.
//! The call is inplace: fills A with its inverse
void cublasInvertMatrix(const CublasMat<float>& A)
{
    static CublasMat<float> Ainv;
    Ainv.resize(A.nrows(), A.ncols(), false);
    cublasInvertMatrix(A, Ainv);
    Ainv >> A;
}

// -------------------------
// Lapack Linear algebra operations

//! Utility call that solves for X the system of linear equations AX=B with a square matrix A
//! It calls the lower level lapackSolveLinearSystem 
//! currently transfers matrices to cpu and calls lapack there 
//! TODO: do a gpu version

template<class T>
inline void solveLinearSystemOnHost(const CublasMat<T>& A, CublasMat<T>& X, const CublasMat<T>& B)
{
    assert(A.nrows() == A.ncols());
    assert(A.nrows() == B.nrows());

    X.resize(B.nrows(), B.ncols());

    static BMat<T> A_cpu;
    static BMat<T> B_cpu;
    static BVec<int> pivots;
    
    A_cpu.resize(A.nrows(), A.ncols());
    A_cpu << A;

    B_cpu.resize(B.nrows(), B.ncols());
    B_cpu << B;

    pivots.resize(A.nrows(),false);

    int info = lapackSolveLinearSystem(A_cpu, B_cpu, pivots);
    if (info!=0)
        PLERROR("lapackSolveLinearSystem returned with error info status of " << info);

    B_cpu >> X;
}

template<class T>
inline void solveLinearSystemOnDevice(const CublasMat<T>& A, CublasMat<T>& X, const CublasMat<T>& B)
{
    assert(A.nrows() == A.ncols());
    assert(A.nrows() == B.nrows());

    // We will use a copy of A because the underlying call cublasSolveLinearSystem does not preserve A
    static CublasMat<T> A_copy;
    A_copy.resize(A.nrows(), A.ncols());
    A_copy << A;

    // resize X if nor already the right size
    X.resize(B.nrows(), B.ncols());
    X << B;

    static CublasVec<int> pivots;
    pivots.resize(A.nrows(), false);  // resize without unnecessarily copying data
    cublasSolveLinearSystem(A_copy, X, pivots);
}

template<class T>
inline void solveLinearSystem(const CublasMat<T>& A, CublasMat<T>& X, const CublasMat<T>& B, bool on_device=false)
{
    if (on_device)
        solveLinearSystemOnDevice(A, X, B);
    else
        solveLinearSystemOnHost(A, X, B);
}


//! This function inverts a matrix in place.
// Currently peforms it on cpu, calling lapack
// TODO: not yet implemented for CublasMat

void invertMatrixOnHost(const CublasMat<real>& A, const CublasMat<real>& Ainv)
{
    // If the matrix is empty, just do nothing instead of crashing.
    if (!A.isEmpty()) 
    {
        int m = A.nrows();
        int n = A.ncols();
        BMat<real> A_cpu(BVec<real>::tmpvec(m*n), m, n);
        // printf(" in invertMatrix, before copy: max_abs_diff = %e \n", max_abs_diff(A_cpu, A));
        A_cpu << A;
        // printf(" in invertMatrix, after copy: max_abs_diff = %e \n", max_abs_diff(A_cpu, A));
        lapackInvertMatrix(A_cpu);
        A_cpu >> Ainv;
    }
}

inline void invertMatrixOnHost(const CublasMat<real>& A)
{
    invertMatrixOnHost(A,A);
}

//! A <- A^-1
//! inplace matrix inversion
void invertMatrix(const CublasMat<real>& A, bool on_device=false)
{
    if (on_device) 
        cublasInvertMatrix(A);        
    else
        invertMatrixOnHost(A);
}

//! Ainv <- A^-1
void invertMatrix(const CublasMat<real>& A, const CublasMat<real>& Ainv, bool on_device=false)
{
    if ( A.data() == Ainv.data() ) // we actually want an inplace inversion
        invertMatrix(A, on_device);
    else
    {
        if (on_device) 
            cublasInvertMatrix(A, Ainv);        
        else
            invertMatrixOnHost(A, Ainv);
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
