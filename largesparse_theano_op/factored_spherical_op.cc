#section support_code
//

#define DO_NOT_INCLUDE_CUDA

#define TFAC
// #define CUDATFAC

#if defined(TFAC)
#include <TFactoredSphericalLayer.h>
#elif defined(CUDATFAC)
#include <cublas_linalg.h>
#include <cublas_k_sparse.h>
#include <TFactoredSphericalLayer.h>
#else
#include <FactoredSphericalLayer.h>
#endif
// #include <test_debug.cc>

#include <stdio.h>
#include <stdexcept>

#define MAX_MSG_LEN 1000
#define PyArray_DATA_comma_STRIDE(M,T) (T*)PyArray_DATA(M), PyArray_STRIDE(M,0)/sizeof(T)

#section support_code_apply

//! checks dimensions of m 
void check_matrix(const char* name, PyArrayObject* m, int nrows, int ncols, const char* extramsg="")
{
  static char msg[MAX_MSG_LEN];

  if (m==0)
    {
      snprintf(msg, MAX_MSG_LEN, "Received null pointer for %s", name);
      throw std::runtime_error(msg);
    }

  // check whether it's a matrix (2d ndarray)
  if (PyArray_NDIM(m) != 2)
    {
      snprintf(msg, MAX_MSG_LEN, "%s must be two-dimensional", name);
      throw std::runtime_error(msg);
    }

  int nr = PyArray_DIM(m,0);
  int nc = PyArray_DIM(m,1);

  // check that unstrided in second dim
  if (PyArray_STRIDE(m,1) != PyArray_ITEMSIZE(m) && nc!=1 ) 
    {
      snprintf(msg, MAX_MSG_LEN, "%s must be unstrided (i.e. have contiguous elements) in its second (i.e. last) dimension", name);
      throw std::runtime_error(  msg);
    }

  // fprintf(stderr, "Matrix %s is %d x %d \n", name, nr, nc);
  if ( nr!=nrows || nc!=ncols )
    {
      snprintf(msg, MAX_MSG_LEN, "%s is expected to be a %d x %d matrix. But its dimensions are %d x %d. %s", name, nrows, ncols, nr, nc, extramsg);
      throw std::runtime_error(msg);
    }  
}

//! checks dimensions of v
void check_vector(const char* name, PyArrayObject* v, int n, const char* extramsg="")
{
  static char msg[MAX_MSG_LEN];

  if (v==0)
    {
      snprintf(msg, MAX_MSG_LEN, "Received null pointer for %s", name);
      throw std::runtime_error(msg);
    }

  // check whether it's a vector (1d ndarray)
  if (PyArray_NDIM(v) != 1)
    {
      snprintf(msg, MAX_MSG_LEN, "%s must be one-dimensional", name);
      throw std::runtime_error(msg);
    }

  int v_n = PyArray_DIM(v,0);
  // fprintf(stderr, "Vector %s is of length %d\n", name, v_n);
  if ( v_n!=n )
    {
      snprintf(msg, MAX_MSG_LEN, "%s is expected to be a vector of length %d. But its length is %d.", name, n, v_n, extramsg);
      throw std::runtime_error(msg);
    }  
}

//! checks we have a 0-dimensional ndarray
void check_scalar(const char* name, PyArrayObject* s, const char* extramsg="")
{
  static char msg[MAX_MSG_LEN];

  if (s==0)
    {
      snprintf(msg, MAX_MSG_LEN, "Received null pointer for %s", name);
      throw std::runtime_error(msg);
    }

  // check whether it's a scalar (0d ndarray)
  if (PyArray_NDIM(s) != 0)
    {
      snprintf(msg, MAX_MSG_LEN, "%s must be a zero-dimensional array (a scalar)", name);
      throw std::runtime_error(msg);
    }
}


//! If *A has not been created yet, or if it doesn't have the specified dimensions, (re)build it 
inline void ensure_output_pyarray_1D(PyArrayObject **A, npy_intp n, int npy_typenum)
{
  if (*A == 0  || PyArray_NDIM(*A)!=1 || PyArray_DIM(*A,0) != n )  
    { 
      // Reference received to invalid output variable.             
      //   Decrease received reference's ref count and allocate new output variable 
      Py_XDECREF(*A); 
      *A = (PyArrayObject*)PyArray_EMPTY(1, &n, npy_typenum, 0);
      if (*A == 0) 
        throw std::runtime_error("Could not allocate storage for 1D array");
    } 
} 

//! If *A has not been created yet, or if it doesn't have the specified dimensions, (re)build it 
inline void ensure_output_pyarray_2D(PyArrayObject **A, npy_intp nrows, npy_intp ncols, int npy_typenum)
{
  // fprintf(stderr, "Ensuring matrix %d x %d\n", nrows, ncols);
  if (*A == 0  || PyArray_NDIM(*A)!=2
      || PyArray_DIM(*A,0) != nrows || PyArray_DIM(*A,1) != ncols )  
    { 
      // Reference received to invalid output variable.             
      //   Decrease received reference's ref count and allocate new output variable 
      Py_XDECREF(*A); 
      npy_intp dims[] = {nrows, ncols};
      *A = (PyArrayObject*)PyArray_EMPTY(2, dims, npy_typenum, 0);
      if (*A == 0) 
        throw std::runtime_error("Could not allocate storage for 2D array");
    } 
} 


template<int invup_mode,
  int stabilize_period, 
  int debug_print>
int APPLY_SPECIFIC(factored_spherical_op_fprop)
  ( // call parameters
   PyArrayObject *V,         // row-major: D x d
   PyArrayObject *UT,        // row-major: d x d
   PyArrayObject *Uinv,      // row-major: d x d
   PyArrayObject *QT,         // row-major: d x d
   PyArrayObject *omega,     // vector of length: d
   PyArrayObject *w_bar,     // vector of length: d
   PyArrayObject *HT,        // row-major: m x d 
   PyArrayObject *KindexesT, // row major: m x K
   // Outputs:
   PyArrayObject **AT,       // row-major: m x K
   PyArrayObject **q,        // vector of length: m
   PyArrayObject **s,        // vector of length: m
   PyArrayObject **work_d,   // row-major: 5m x d
   PyArrayObject **work_m   // row-major: (3+2m) x m     
    ) 
{

  try {

  int D = PyArray_DIM(V,0);
  int d = PyArray_DIM(V,1);
  int m = PyArray_DIM(HT,0);
  int K = PyArray_DIM(KindexesT,1);

  // Normally dtypes for the matrices have been checked in the python code already,
  // KindexesT is supposed to have type int32

  // Check dimensions
  check_matrix("V", V, D, d);
  check_matrix("UT", UT, d, d);
  check_matrix("Uinv", Uinv, d, d);
  check_matrix("QT", QT, d, d);
  check_vector("omega", omega, d);
  check_vector("w_bar", w_bar, d);
  check_matrix("HT", HT, m, d );
  check_matrix("KindexesT", KindexesT, m, K);
  
  // Note that all real valued variables are supposed to be of the same real
  // dtype (either all float32 or all float64). This is checked in the op's python make_node function.
  // So here we will use DTYPE_INPUT_0 as defining that type for reals, for all real-valued variables.
  // and similarly use ITEMSIZE_INPUT_0 for its byte size, and similarly PyArray_TYPE(V) for the numpy typenum for reals

  int npy_real_typenum = PyArray_TYPE(V);
  ensure_output_pyarray_2D(AT, m, K, npy_real_typenum);
  ensure_output_pyarray_1D(q, m, npy_real_typenum);
  ensure_output_pyarray_1D(s, m, npy_real_typenum);

  ensure_output_pyarray_2D(work_d, 5*m, d, npy_real_typenum);
  ensure_output_pyarray_2D(work_m, (3+2*m), m, npy_real_typenum);

  //  printf("use_lower=%d, invup_mode=%d\n", use_lower, invup_mode);
  int use_lower = 1;  // for now always use lower (as full is not implemented)

  // Note that the call expects column-major matrices.
  // So we pass it row-major versions of their *transpose*
  // (ex: we pas a row-major UT for column-major parameter U
  // and we pass row-majour Uinv for column major parameter UinvT ))
  // This changes nothing for matrices that are supposed to be symmetric (s.a. Q).

#if defined(TFAC)
    PLearn::TFactoredSphericalLayer< BMat<DTYPE_INPUT_0>, BMat<int> >::stateless_fprop
#elif defined(CUDATFAC)
    PLearn::TFactoredSphericalLayer< CublasMat<DTYPE_INPUT_0>, BMat<int> >::stateless_fprop
#else
  PLearn::stateless_fprop_FactoredSphericalLayer
#endif                                                 
    (d, D, m, K,
             // model parameters and bookkeeping variables
             PyArray_DATA_comma_STRIDE(V, DTYPE_INPUT_0),
             PyArray_DATA_comma_STRIDE(UT, DTYPE_INPUT_0),
             PyArray_DATA_comma_STRIDE(Uinv, DTYPE_INPUT_0),
             PyArray_DATA_comma_STRIDE(QT, DTYPE_INPUT_0),
             PyArray_DATA_comma_STRIDE(omega, DTYPE_INPUT_0),
             PyArray_DATA_comma_STRIDE(w_bar, DTYPE_INPUT_0),

             // workspace matrices
             PyArray_DATA_comma_STRIDE(*work_d, DTYPE_INPUT_0),
             PyArray_DATA_comma_STRIDE(*work_m, DTYPE_INPUT_0),

             // minibatch value inputs
             PyArray_DATA_comma_STRIDE(HT, DTYPE_INPUT_0),
             PyArray_DATA_comma_STRIDE(KindexesT, int),
             
             // minibatch value outputs:
             PyArray_DATA_comma_STRIDE(*AT, DTYPE_INPUT_0),
             PyArray_DATA_comma_STRIDE(*q, DTYPE_INPUT_0),
             PyArray_DATA_comma_STRIDE(*s, DTYPE_INPUT_0),
             
             // extra control parameters
             use_lower,
             debug_print
             );

  } // end of try block

  catch (const std::exception& ex) 
    {
      PyErr_SetString(PyExc_ValueError, ex.what());
      return 1;
    }

  return 0;

}



template<int invup_mode,
  int stabilize_period, 
  int debug_print>
int APPLY_SPECIFIC(factored_spherical_op_bprop_update)
  ( // call parameters
   PyArrayObject *V,         // row-major: D x d
   PyArrayObject *UT,        // row-major: d x d
   PyArrayObject *Uinv,      // row-major: d x d
   PyArrayObject *QT,         // row-major: d x d
   PyArrayObject *omega,     // vector of length: d
   PyArrayObject *w_bar,     // vector of length: d
   PyArrayObject *HT,        // row-major: m x d 
   PyArrayObject *KindexesT, // row major: m x K

   PyArrayObject *work_d,   // row-major: 5m x d
   PyArrayObject *work_m,   // row-major: (3+2m) x m     

   PyArrayObject *grad_AT,       // row-major: m x K
   PyArrayObject *grad_q,        // vector of length: m
   PyArrayObject *grad_s,         // vector of length: m
   PyArrayObject *eta,         // scalar learning rate
   // output
   PyArrayObject **grad_HT        // row-major: m x d 
    ) 
{
  static unsigned long nupdates = 0;

  try {

  int D = PyArray_DIM(V,0);
  int d = PyArray_DIM(V,1);
  int m = PyArray_DIM(HT,0);
  int K = PyArray_DIM(KindexesT,1);

  // Normally dtypes for the matrices have been checked in the python code already,
  // Kindexes is supposed to have type int32

  // Check dimensions
  check_matrix("V", V, D, d);
  check_matrix("UT", UT, d, d);
  check_matrix("Uinv", Uinv, d, d);
  check_matrix("QT", QT, d, d);
  check_vector("omega", omega, d);
  check_vector("w_bar", w_bar, d);
  check_matrix("HT", HT, m, d);
  check_matrix("KindexesT", KindexesT, m, K);

  // check_matrix("work_d", work_d, 5*m, d);
  check_matrix("OHO work_m", work_m, 3+2*m, m);

  check_matrix("grad_AT", grad_AT, m, K);
  check_vector("grad_q", grad_q, m);
  check_vector("grad_s", grad_s, m);
  check_scalar("eta", eta);
  
  // Note that all real valued variables are supposed to be of the same real
  // dtype (either all float32 or all float64). This is checked in the op's python make_node function.
  // So here we will use DTYPE_INPUT_0 as defining that type for reals, for all real-valued variables.
  // and similarly use ITEMSIZE_INPUT_0 for its byte size, and similarly PyArray_TYPE(V) for the numpy typenum for reals

  int npy_real_typenum = PyArray_TYPE(V);
  ensure_output_pyarray_2D(grad_HT, m, d, npy_real_typenum);

  DTYPE_INPUT_0 eta_val =  *((DTYPE_INPUT_0*) PyArray_DATA(eta)); // learning rate
  
  // Note that the call expects column-major matrices.
  // So we pass it row-major versions of their *transpose*
  // (ex: we pas a row-major UT for column-major parameter U
  // and we pass row-majour Uinv for column major parameter UinvT ))
  // This changes nothing for matrices that are supposed to be symmetric (s.a. Q).


  int accumulate_gradients_in_grad_H = 0; // do not accumulate, but ovrwrite grad_H with the gradient
  int use_lower = 1;  // for now always use lower (as full is not implemented)
  
  // Perform numerical stability control?
  int run_stability_control = 0;
  if (eta>0)
    {
      nupdates++;
      if (nupdates % stabilize_period == 0)
        run_stability_control = 1;
    }       
  
#if defined(TFAC)
    PLearn::TFactoredSphericalLayer< BMat<DTYPE_INPUT_0>, BMat<int> >::stateless_bprop_update
#elif defined(CUDATFAC)
    PLearn::TFactoredSphericalLayer< CublasMat<DTYPE_INPUT_0>, BMat<int> >::stateless_bprop_update
#else
  PLearn::stateless_bprop_update_FactoredSphericalLayer
#endif                                                 
    (d, D, m, K,
             // model parameters and bookkeeping variables
             PyArray_DATA_comma_STRIDE(V, DTYPE_INPUT_0),
             PyArray_DATA_comma_STRIDE(UT, DTYPE_INPUT_0),
             PyArray_DATA_comma_STRIDE(Uinv, DTYPE_INPUT_0),
             PyArray_DATA_comma_STRIDE(QT, DTYPE_INPUT_0),
             PyArray_DATA_comma_STRIDE(omega, DTYPE_INPUT_0),
             PyArray_DATA_comma_STRIDE(w_bar, DTYPE_INPUT_0),
             // workspace
             PyArray_DATA_comma_STRIDE(work_d, DTYPE_INPUT_0),
             PyArray_DATA_comma_STRIDE(work_m, DTYPE_INPUT_0),             
             // minibatch value inputs
             PyArray_DATA_comma_STRIDE(HT, DTYPE_INPUT_0),
             PyArray_DATA_comma_STRIDE(KindexesT, int),
             // minibatch gradient inputs
             PyArray_DATA_comma_STRIDE(grad_AT, DTYPE_INPUT_0),
             PyArray_DATA_comma_STRIDE(grad_q, DTYPE_INPUT_0),
             PyArray_DATA_comma_STRIDE(grad_s, DTYPE_INPUT_0),

             // minibatch gradient output
             PyArray_DATA_comma_STRIDE(*grad_HT, DTYPE_INPUT_0),

             // bprop and update control parameters
             eta_val,
             accumulate_gradients_in_grad_H,
             use_lower,
             invup_mode,
             debug_print,
             run_stability_control
             );

  } // end of try block

  catch (const std::exception& ex) 
    {
      PyErr_SetString(PyExc_ValueError, ex.what());
      return 1;
    }

  return 0;

}

