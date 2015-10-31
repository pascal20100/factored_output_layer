#section support_code
//

#define DO_NOT_INCLUDE_CUDA

#include <blas_linalg.h>
//#include "../largesparse_cpu.cc"
//#include "../test_debug.cc"
#include <largesparse_cpu.cc>
// #include <test_debug.cc>

#include <stdio.h>
#include <stdexcept>

#define MAX_MSG_LEN 1000
#define PyArray_DATA_comma_STRIDE(M,T) (T*)PyArray_DATA(M), PyArray_STRIDE(M,0)/sizeof(T)

BMat<float> transposed_as_BMat_float(PyArrayObject* m, const char* name="")
{
  char msg[MAX_MSG_LEN];
 
  // check whether it's a matrix (2d ndarray)
  if (PyArray_NDIM(m) != 2)
    {
      snprintf(msg, MAX_MSG_LEN, "In transposed_as_BMat_float(%s) must be two-dimensional", name);
      throw std::runtime_error(  msg);
    }
// adsfe
  // check if elements are int
  if (PyArray_TYPE(m) != NPY_FLOAT32)
    {
      snprintf(msg, MAX_MSG_LEN, "In transposed_as_BMat_float(%s) must have elements of type int", name);
      throw std::runtime_error(  msg);
    }

  // check if unstrided in second dim
  if (PyArray_STRIDE(m,1) != sizeof(float) ) 
    {
             snprintf(msg, MAX_MSG_LEN, "In transposed_as_BMat_float(%s) must be unstrided (i.e. have contiguous elements) in its second (i.e. last) dimension", name);
      throw std::runtime_error(  msg);
    }

  return BMat<float>( (float*) PyArray_DATA(m),
                      PyArray_DIM(m,1), PyArray_DIM(m,0),
                      PyArray_STRIDE(m,0) / sizeof(float)
                      );
}

  //! checks whether it's a valid matrix (2d ndarray, unstrided in its second dim), with int elements
void check_int_matrix(const char* name, PyArrayObject* m)
{
  static char msg[MAX_MSG_LEN];
 
  // check whether it's a matrix (2d ndarray)
  if (PyArray_NDIM(m) != 2)
    {
      snprintf(msg, MAX_MSG_LEN, "%s must be two-dimensional", name);
      throw std::runtime_error(  msg);
    }

  // check if elements are int
  if (PyArray_TYPE(m) != NPY_INT32)
    {
      snprintf(msg, MAX_MSG_LEN, "%s must have elements of type int", name);
      throw std::runtime_error(  msg);
    }

  // check if unstrided in second dim
  if (PyArray_STRIDE(m,1) != sizeof(int) ) 
    {
             snprintf(msg, MAX_MSG_LEN, "%s must be unstrided (i.e. have contiguous elements) in its second (i.e. last) dimension", name);
      throw std::runtime_error(  msg);
    }    

} 

//! checks whether it's a valid matrix (2d ndarray, unstrided in its second dim), with either float or double elements
void check_real_matrix(const char* name, PyArrayObject* m)
{
  static char msg[MAX_MSG_LEN];
 
  // check whether it's a matrix (2d ndarray)
  if (PyArray_NDIM(m) != 2)
    {
      snprintf(msg, MAX_MSG_LEN, "%s must be two-dimensional", name);
      throw std::runtime_error(  msg);
    }

  // determin elemsize and check whether it's a float or double
  int elemsize = 0;
  switch(PyArray_TYPE(m))
    {
    case NPY_FLOAT64:
      elemsize = sizeof(double);
      break;
    case NPY_FLOAT32:
      elemsize = sizeof(float);
      break;
    default:
      snprintf(msg, MAX_MSG_LEN, "%s must have elements of type double or float", name);
      throw std::runtime_error(  msg);
    }      

  // check if unstrided in second dim
  if (PyArray_STRIDE(m,1) != elemsize ) 
    {
      snprintf(msg, MAX_MSG_LEN, "%s must be unstrided (i.e. have contiguous elements) in its second (i.e. last) dimension", name);
      throw std::runtime_error(  msg);
    }    
  
  // all checks correct
} 

//! checks dimensions of m 
void check_matrix_dimensions(const char* name, PyArrayObject* m, int nrows, int ncols, const char* extramsg="")
{
  static char msg[MAX_MSG_LEN];

  int nr = PyArray_DIM(m,0);
  int nc = PyArray_DIM(m,1);
  if ( nr!=nrows || nc!=ncols )
    {
      snprintf(msg, MAX_MSG_LEN, "%s is expected to be a %d x %d matrix. But its dimensions are %d x %d. %s", name, nrows, ncols, nr, nc, extramsg);
      throw std::runtime_error(  msg);
    }  
}


/*
  Matrix dimensions (for row-major matrices):
  V: D x d
  U: d x d
  UinvT: d x d
  Q: d x d
  H: m x d
  Y_indexes: m x K
  Y_values: m x K

 */

void largesparse_check_all_dimensions(PyArrayObject *V,
                                      PyArrayObject *U,
                                      PyArrayObject *UinvT, 
                                      PyArrayObject *Q, 
                                      PyArrayObject *H, 
                                      PyArrayObject *Y_indexes,
                                      PyArrayObject *Y_values, 
                                      PyArrayObject *learning_rate,
                                      PyArrayObject **cost,
                                      PyArrayObject **grad_H) 
{

  check_real_matrix("V", V);
  check_real_matrix("U", U);
  check_real_matrix("UinvT", UinvT);
  check_real_matrix("Q", Q);
  check_real_matrix("H", H);
  check_int_matrix("Y_indexes", Y_indexes);
  check_real_matrix("Y_values", Y_values);
//  printf("V dims: %d %d  strides in bytes: %d %d \n",PyArray_DIM(V,0), PyArray_DIM(V,1), (int)PyArray_STRIDE(V,0), PyArray_STRIDE(V,1));
  
  // V is expected to be a D x d row-major (which amounts to d x D column-major).
  int D = PyArray_DIM(V,0);
  int d = PyArray_DIM(V,1);
  
//  printf("Y_indexes dims: %d %d  strides in bytes: %d %d \n",PyArray_DIM(Y_indexes,0), PyArray_DIM(Y_indexes,1), (int)PyArray_STRIDE(Y_indexes,0), PyArray_STRIDE(Y_indexes,1));
  
  // U, UinvT, Q, are supposed to be d x d matrices
  check_matrix_dimensions("U", U, d, d, " (expected dimension determined from V)");
  check_matrix_dimensions("UinvT", UinvT, d, d, " (expected dimension determined from V)");
  check_matrix_dimensions("Q", Q, d, d, " (expected dimension determined from V)");
  
  // Y_indexes is expected to be a m x K row-major matrix (which amounts to a K x m column-major matrix) 
  // and Y_values is suppose to be of the same dimension
  int m = PyArray_DIM(Y_indexes,0);
  int K = PyArray_DIM(Y_indexes,1);
  
  check_matrix_dimensions("Y_values", Y_values, m, K, " (dimensions should match those of from Y_indexes)");
  
  // H is supposed to be a m x d row-major matrix (which amounbts to d x m column-major matrix)
  check_matrix_dimensions("H", H, m, d, " (expected dimensions were determined from number of rows of Y_indexes and number of columns of V)");
  
  // grad_H and cost may be allocated with the right size only later. So we don't check them here. 
  // grad_H is supposed to be the same
  /*
    if ( grad_H != 0 )
    {
    check_matrix_dimensions("grad_H", *grad_H, m, d, " (expected dimensions were determined from number of rows of Y_indexes and number of columns of V)");  
    }
  */
  
}



#section support_code_apply


int APPLY_SPECIFIC(largesparsetargets_maintainance)(PyArrayObject *maintainance_op,
                                                    PyArrayObject *V,
                                                    PyArrayObject *U,
                                                    PyArrayObject *UinvT, 
                                                    PyArrayObject *Q)
{
  try {


    // largesparse_check_all_dimensions( V, U, UinvT, Q, H, Y_indexes, Y_values, learning_rate, cost, grad_H); 
  
    int maintainance_op_val = * ((DTYPE_INPUT_0*) PyArray_DATA(maintainance_op));

    int D = PyArray_DIM(V,0);
    int d = PyArray_DIM(V,1);
    
  // Normally dtypes for the matrices have been checked in the python code already,
  // Y_indexes is supposed to have type int32

  // Note that all real valued variables are supposed to be of the same real
  // dtype (either all float32 or all float64). This is checked in the op's python make_node function.
  // So here we will use DTYPE_INPUT_1 as defining tha type for reals, for all real-valued variables.
  // (and similarly use ITEMSIZE_INPUT_1 for its byte size)

    
  // Compute mode flags
  bool use_Qtilde = false;
  bool use_lower = true;
  int invup_mode = 0;


  largesparse_cpu_maintainance(maintainance_op_val,
                               d, D,
                               PyArray_DATA_comma_STRIDE(V, DTYPE_INPUT_1),
                               PyArray_DATA_comma_STRIDE(U, DTYPE_INPUT_1),
                               PyArray_DATA_comma_STRIDE(UinvT, DTYPE_INPUT_1),
                               PyArray_DATA_comma_STRIDE(Q, DTYPE_INPUT_1),
                               use_Qtilde, use_lower, invup_mode);

    }

  catch (const std::exception& ex) 
    {
      PyErr_SetString(PyExc_ValueError, ex.what());
      return 1;
    }

  return 0;

}




// Version outputting both the scalar cost and grad_H
int APPLY_SPECIFIC(largesparsetargets)(PyArrayObject *V,
                                       PyArrayObject *U,
                                       PyArrayObject *UinvT, 
                                       PyArrayObject *Q, 
                                       PyArrayObject *H, 
                                       PyArrayObject *Y_indexes,
                                       PyArrayObject *Y_values, 
                                       PyArrayObject *learning_rate,
                                       PyArrayObject *py_use_qtilde, 
                                       PyArrayObject *py_use_lower, 
                                       PyArrayObject *py_invup_mode,
                                       PyArrayObject *py_stabilize_period, 
                                       PyArrayObject *py_unfactorize_period,
                                       PyArrayObject *py_debug_print,
                                       PyArrayObject **cost,
                                       PyArrayObject **grad_H) 
{

  try {


    //printf("On essaye directement une conversion en BMat\n");
    //BMat<float> V_ = transposed_as_BMat_float(V, "V");


  largesparse_check_all_dimensions( V, U, UinvT, Q, H, Y_indexes, Y_values, learning_rate, cost, grad_H); 
  int D = PyArray_DIM(V,0);
  int d = PyArray_DIM(V,1);
  int m = PyArray_DIM(Y_indexes,0);
  int K = PyArray_DIM(Y_indexes,1);

  // First properly set up outputs grad_H and cost

  // Normally dtypes for the matrices have been checked in the python code already,
  // Y_indexes is supposed to have type int32

  // Note that all real valued variables are supposed to be of the same real
  // dtype (either all float32 or all float64). This is checked in the op's python make_node function.
  // So here we will use DTYPE_INPUT_0 as defining tha type for reals, for all real-valued variables.
  // (and similarly use ITEMSIZE_INPUT_0 for its byte size)


  DTYPE_INPUT_0* grad_H_data = 0;
  int grad_H_stride = 0;  
  if (grad_H != 0) // we will want to compute grad_H
    {
      if (*grad_H == 0
          || PyArray_DIM(*grad_H,0) != PyArray_DIM(H,0)
          || PyArray_DIM(*grad_H,1) != PyArray_DIM(H,1) )
        
        {
          // Reference received to invalid output variable.
          //   Decrease received reference's ref count and allocate new output variable 
          Py_XDECREF(*grad_H);
          *grad_H = (PyArrayObject *)PyArray_NewLikeArray(H, NPY_ANYORDER, NULL, 0);          
          // *grad_H = (PyArrayObject*)PyArray_EMPTY(2, PyArray_DIMS(H), NPY_FLOAT32, 0);
          if (*grad_H == 0) 
            throw std::runtime_error("Could not allocate storage for grad_H");
        } 
      grad_H_data = (DTYPE_INPUT_0*) PyArray_DATA(*grad_H);
      grad_H_stride = PyArray_STRIDE(*grad_H,0) / ITEMSIZE_INPUT_0;
    }

  DTYPE_INPUT_0* cost_data = 0;
  if (cost != 0)
    {
      if (*cost == 0)
        {
          // Reference received to invalid output variable.
          //   Decrease received reference's ref count and allocate new output variable 
          Py_XDECREF(*cost);
          // *cost = (PyArrayObject *)PyArray_NewLikeArray(learning_rate, NPY_ANYORDER, NULL, 0);
                    // *cost = (PyArrayObject *)PyArray_NewLikeArray(learning_rate, NPY_ANYORDER, NULL, 0);
          *cost = (PyArrayObject*)PyArray_EMPTY(0, 0, PyArray_TYPE(V), 0);
          if (*cost == 0) 
            throw std::runtime_error("Could not allocate storage for cost");
        }
      cost_data = (DTYPE_INPUT_0*) PyArray_DATA(*cost);
    }

  DTYPE_INPUT_0 lr = 0; // default value means we won't do any update
  if (learning_rate!=0)
  {
    if (PyArray_NDIM(learning_rate) != 0) {
      throw std::runtime_error("learning_rate must be a scalar");}
    lr = * ((DTYPE_INPUT_0*) PyArray_DATA(learning_rate)); // learning rate
  }

  // Compute mode flags
  bool use_Qtilde = (bool) *((DTYPE_INPUT_8*) PyArray_DATA(py_use_qtilde));
  bool use_lower = (bool) *((DTYPE_INPUT_9*) PyArray_DATA(py_use_lower));
  int invup_mode =  *((DTYPE_INPUT_10*)PyArray_DATA(py_invup_mode));
  int stabilize_period = *((DTYPE_INPUT_11*) PyArray_DATA(py_stabilize_period));
  int unfactorize_period = *((DTYPE_INPUT_12*) PyArray_DATA(py_unfactorize_period)); 
  int debug_print = *((DTYPE_INPUT_13*) PyArray_DATA(py_debug_print));

//  printf("use_Qtilde=%d, use_lower=%d, invup_mode=%d\n", use_Qtilde, use_lower, invup_mode);


  largesparse_cpu_fbprop_update(d,
                                D,
                                K,
                                m, 
                                PyArray_DATA_comma_STRIDE(V, DTYPE_INPUT_0),
                                PyArray_DATA_comma_STRIDE(U, DTYPE_INPUT_0),
                                PyArray_DATA_comma_STRIDE(UinvT, DTYPE_INPUT_0),
                                PyArray_DATA_comma_STRIDE(Q, DTYPE_INPUT_0),
                                PyArray_DATA_comma_STRIDE(H, DTYPE_INPUT_0),
                                PyArray_DATA_comma_STRIDE(Y_indexes, int),
                                PyArray_DATA_comma_STRIDE(Y_values, DTYPE_INPUT_0),
                                lr,
                                cost_data, 
                                grad_H_data, grad_H_stride,
                                use_Qtilde, use_lower, invup_mode,
                                stabilize_period,
                                unfactorize_period,
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


// Version outputting only the cost, but not grad_H
int APPLY_SPECIFIC(largesparsetargets_0)(PyArrayObject *V,
                                         PyArrayObject *U,
                                         PyArrayObject *UinvT, 
                                         PyArrayObject *Q, 
                                         PyArrayObject *H, 
                                         PyArrayObject *Y_indexes,
                                         PyArrayObject *Y_values, 
                                         PyArrayObject *learning_rate,
                                         PyArrayObject *use_qtilde, 
                                         PyArrayObject *use_lower, 
                                         PyArrayObject *invup_mode,
                                         PyArrayObject *stabilize_period, 
                                         PyArrayObject *unfactorize_period,
                                         PyArrayObject *debug_print,
                                         PyArrayObject **cost)

{
   return APPLY_SPECIFIC(largesparsetargets)(V, U, UinvT, Q, H, Y_indexes, Y_values, 0,
                                             use_qtilde, use_lower, invup_mode, 
                                             stabilize_period, unfactorize_period, 
                                             debug_print, cost, 0);
}


// Version outputting only grad_H but not the cost
int APPLY_SPECIFIC(largesparsetargets_1)(PyArrayObject *V,
                                         PyArrayObject *U,
                                         PyArrayObject *UinvT, 
                                         PyArrayObject *Q, 
                                         PyArrayObject *H, 
                                         PyArrayObject *Y_indexes,
                                         PyArrayObject *Y_values, 
                                         PyArrayObject *learning_rate,
                                         PyArrayObject *use_qtilde, 
                                         PyArrayObject *use_lower, 
                                         PyArrayObject *invup_mode,
                                         PyArrayObject *stabilize_period, 
                                         PyArrayObject *unfactorize_period,
                                         PyArrayObject *debug_print,
                                         PyArrayObject **grad_H)

{
  return APPLY_SPECIFIC(largesparsetargets)(V, U, UinvT, Q, H, Y_indexes, Y_values, learning_rate,
                                            use_qtilde, use_lower, invup_mode, 
                                            stabilize_period, unfactorize_period, 
                                            debug_print, 0, grad_H);
}

