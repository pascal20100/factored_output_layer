// -*- C++ -*-

// blas_proto_overload.h
//
// Copyright (C) 2014 Pascal Vincent
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
// 

#ifndef blas_proto_overload_INC
#define blas_proto_overload_INC

// #include <plearn/math/lapack_proto.h>


extern "C" { 

void xerbla_(char*, void *);

/***********/
/* Level 1 */
/***********/

/* Single Precision */
    float sdot_(const int* n, const float* x, const int* incx, const float* y, const int* incy);
    void srot_(const int*, float *, const int*, float *, const int*, const float *, const float *);
    void srotg_(float *,float *,float *,float *);    
    void srotm_( const int*, float *, const int*, float *, const int*, const float *);
    void srotmg_(float *,float *,float *,const float *, float *);
    void sswap_( const int*, float *, const int*, float *, const int*);
    void scopy_( const int*, const float *, const int*, float *, const int*);
    void saxpy_( const int* n, const float* alpha, const float *x, const int* incx, float *y, const int* incy);
    void sdot_sub_(const int*, const float *, const int*, const float *, const int*, float *);
    void sdsdot_sub_( const int*, const float *, const float *, const int*, const float *, const int*, float *);
    void sscal_( const int* n, const float *alpha, float *x, const int* incx);
    void snrm2_sub_( const int*, const float *, const int*, float *);
    void sasum_sub_( const int*, const float *, const int*, float *);
    void isamax_sub_( const int*, const float * , const int*, const int*);

/* Double Precision */

    double ddot_(const int* n, const double* x, const int* incx, const double* y, const int* incy);
    void drot_(const int*, double *, const int*, double *, const int*, const double *, const double *);
    void drotg_(double *,double *,double *,double *);    
    void drotm_( const int*, double *, const int*, double *, const int*, const double *);
    void drotmg_(double *,double *,double *,const double *, double *);
    void dswap_( const int*, double *, const int*, double *, const int*);
    void dcopy_( const int*, const double *, const int*, double *, const int*);
    void daxpy_( const int* n, const double *alpha, const double *x, const int* incx, double *y, const int* incy);
    void dswap_( const int*, double *, const int*, double *, const int*);
    void dsdot_sub_(const int*, const float *, const int*, const float *, const int*, double *);
    void ddot_sub_( const int*, const double *, const int*, const double *, const int*, double *);
    void dscal_( const int* n, const double *alpha, double *x, const int* incx);
    void dnrm2_sub_( const int*, const double *, const int*, double *);
    void dasum_sub_( const int*, const double *, const int*, double *);
    void idamax_sub_( const int*, const double * , const int*, const int*);

/* Single Complex Precision */

    void cswap_( const int*, void *, const int*, void *, const int*);
    void ccopy_( const int*, const void *, const int*, void *, const int*);
    void caxpy_( const int*, const void *, const void *, const int*, void *, const int*);
    void cswap_( const int*, void *, const int*, void *, const int*);
    void cdotc_sub_( const int*, const void *, const int*, const void *, const int*, void *);
    void cdotu_sub_( const int*, const void *, const int*, const void *, const int*, void *);
    void cscal_( const int*, const void *, void *, const int*);
    void icamax_sub_( const int*, const void *, const int*, const int*);
    void csscal_( const int*, const float *, void *, const int*);
    void scnrm2_sub_( const int*, const void *, const int*, float *);
    void scasum_sub_( const int*, const void *, const int*, float *);

/* Double Complex Precision */

    void zswap_( const int*, void *, const int*, void *, const int*);
    void zcopy_( const int*, const void *, const int*, void *, const int*);
    void zaxpy_( const int*, const void *, const void *, const int*, void *, const int*);
    void zswap_( const int*, void *, const int*, void *, const int*);
    void zdotc_sub_( const int*, const void *, const int*, const void *, const int*, void *);
    void zdotu_sub_( const int*, const void *, const int*, const void *, const int*, void *);
    void zdscal_( const int*, const double *, void *, const int*);
    void zscal_( const int*, const void *, void *, const int*);
    void dznrm2_sub_( const int*, const void *, const int*, double *);
    void dzasum_sub_( const int*, const void *, const int*, double *);
    void izamax_sub_( const int*, const void *, const int*, const int*);

/***********/
/* Level 2 */
/***********/

/* Single Precision */

    void sgemv_(char* trans, const int* m, const int* n, const float *alpha, const float *A, const int* ldA, const float *x, const int* incx, const float *beta, float *y, const int* incy);
    void sgbmv_(char*, const int*, const int*, const int*, const int*, const float *,  const float *, const int*, const float *, const int*, const float *, float *, const int*);
    void ssymv_(char* uplo, const int* n, const float *alpha, const float *A, const int* ldA, const float *x,  const int* incx, const float *beta, float *y, const int* incy);
    void ssbmv_(char*, const int*, const int*, const float *, const float *, const int*, const float *, const int*, const float *, float *, const int*);
    void sspmv_(char*, const int*, const float *, const float *, const float *, const int*, const float *, float *, const int*);
    void strmv_( char*, char*, char*, const int*, const float *, const int*, float *, const int*);
    void stbmv_( char*, char*, char*, const int*, const int*, const float *, const int*, float *, const int*);
    void strsv_( char*, char*, char*, const int*, const float *, const int*, float *, const int*);
    void stbsv_( char*, char*, char*, const int*, const int*, const float *, const int*, float *, const int*);
    void stpmv_( char*, char*, char*, const int*, const float *, float *, const int*);
    void stpsv_( char*, char*, char*, const int*, const float *, float *, const int*);
    void sger_( const int* m, const int* n, const float *alpha, const float *x, const int* incx, const float *y, const int* incy, float *A, const int* ldA);
    void ssyr_(char* uplo, const int* n, const float* alpha, const float *x, const int* incx, float *A, const int* ldA);
    void sspr_(char*, const int*, const float *, const float *, const int*, float *); 
    void sspr2_(char*, const int*, const float *, const float *, const int*, const float *, const int*,  float *); 
    void ssyr2_(char* uplo, const int* n, const float *alpha, const float *x, const int* incx, const float *y, const int* incy,  float *A, const int* ldA);

/* Double Precision */

    void dgemv_(char* trans, const int* m, const int* n, const double *alpha, const double *A, const int* ldA, const double *x, const int* incx, const double *beta, double *y, const int* incy);
    void dgbmv_(char*, const int*, const int*, const int*, const int*, const double *,  const double *, const int*, const double *, const int*, const double *, double *, const int*);
    void dsymv_(char* uplo, const int* n, const double *alpha, const double *A, const int* ldA, const double *x,  const int* incx, const double *beta, double *y, const int* incy);
    void dsbmv_(char*, const int*, const int*, const double *, const double *, const int*, const double *, const int*, const double *, double *, const int*);
    void dspmv_(char*, const int*, const double *, const double *, const double *, const int*, const double *, double *, const int*);
    void dtrmv_( char*, char*, char*, const int*, const double *, const int*, double *, const int*);
    void dtbmv_( char*, char*, char*, const int*, const int*, const double *, const int*, double *, const int*);
    void dtrsv_( char*, char*, char*, const int*, const double *, const int*, double *, const int*);
    void dtbsv_( char*, char*, char*, const int*, const int*, const double *, const int*, double *, const int*);
    void dtpmv_( char*, char*, char*, const int*, const double *, double *, const int*);
    void dtpsv_( char*, char*, char*, const int*, const double *, double *, const int*);
    void dger_( const int* m, const int* n, const double *alpha, const double *x, const int* incx, const double *y, const int* incy, double *A, const int* ldA);
    void dsyr_(char* uplo, const int* n, const double* alpha, const double *x, const int* incx, double *A, const int* ldA);

    void dspr_(char*, const int*, const double *, const double *, const int*, double *); 
    void dspr2_(char*, const int*, const double *, const double *, const int*, const double *, const int*,  double *); 
    void dsyr2_(char* uplo, const int* n, const double *alpha, const double *x, const int* incx, const double *y, const int* incy,  double *A, const int* ldA);

/* Single Complex Precision */

    void cgemv_(char*, const int*, const int*, const void *, const void *, const int*, const void *, const int*, const void *, void *, const int*);
    void cgbmv_(char*, const int*, const int*, const int*, const int*, const void *,  const void *, const int*, const void *, const int*, const void *, void *, const int*);
    void chemv_(char*, const int*, const void *, const void *, const int*, const void *, const int*, const void *, void *, const int*);
    void chbmv_(char*, const int*, const int*, const void *, const void *, const int*, const void *, const int*, const void *, void *, const int*);
    void chpmv_(char*, const int*, const void *, const void *, const void *, const int*, const void *, void *, const int*);
    void ctrmv_( char*, char*, char*, const int*, const void *, const int*, void *, const int*);
    void ctbmv_( char*, char*, char*, const int*, const int*, const void *, const int*, void *, const int*);
    void ctpmv_( char*, char*, char*, const int*, const void *, void *, const int*);
    void ctrsv_( char*, char*, char*, const int*, const void *, const int*, void *, const int*);
    void ctbsv_( char*, char*, char*, const int*, const int*, const void *, const int*, void *, const int*);
    void ctpsv_( char*, char*, char*, const int*, const void *, void *,const int*);
    void cgerc_( const int*, const int*, const void *, const void *, const int*, const void *, const int*, void *, const int*);
    void cgeru_( const int*, const int*, const void *, const void *, const int*, const void *, const int*, void *,  const int*);
    void cher_(char*, const int*, const float *, const void *, const int*, void *, const int*);
    void cher2_(char*, const int*, const void *, const void *, const int*, const void *, const int*, void *, const int*);
    void chpr_(char*, const int*, const float *, const void *, const int*, void *);
    void chpr2_(char*, const int*, const float *, const void *, const int*, const void *, const int*, void *);

/* Double Complex Precision */

    void zgemv_(char*, const int*, const int*, const void *, const void *, const int*, const void *, const int*, const void *, void *, const int*);
    void zgbmv_(char*, const int*, const int*, const int*, const int*, const void *,  const void *, const int*, const void *, const int*, const void *, void *, const int*);
    void zhemv_(char*, const int*, const void *, const void *, const int*, const void *, const int*, const void *, void *, const int*);
    void zhbmv_(char*, const int*, const int*, const void *, const void *, const int*, const void *, const int*, const void *, void *, const int*);
    void zhpmv_(char*, const int*, const void *, const void *, const void *, const int*, const void *, void *, const int*);
    void ztrmv_( char*, char*, char*, const int*, const void *, const int*, void *, const int*);
    void ztbmv_( char*, char*, char*, const int*, const int*, const void *, const int*, void *, const int*);
    void ztpmv_( char*, char*, char*, const int*, const void *, void *, const int*);
    void ztrsv_( char*, char*, char*, const int*, const void *, const int*, void *, const int*);
    void ztbsv_( char*, char*, char*, const int*, const int*, const void *, const int*, void *, const int*);
    void ztpsv_( char*, char*, char*, const int*, const void *, void *,const int*);
    void zgerc_( const int*, const int*, const void *, const void *, const int*, const void *, const int*, void *, const int*);
    void zgeru_( const int*, const int*, const void *, const void *, const int*, const void *, const int*, void *,  const int*);
    void zher_(char*, const int*, const double *, const void *, const int*, void *, const int*);
    void zher2_(char*, const int*, const void *, const void *, const int*, const void *, const int*, void *, const int*);
    void zhpr_(char*, const int*, const double *, const void *, const int*, void *);
    void zhpr2_(char*, const int*, const double *, const void *, const int*, const void *, const int*, void *);

/***********/
/* Level 3 */
/***********/

/* Single Precision */

    void sgemm_(char* transA, char* transB, const int* m, const int* n, const int* k, 
                const float *alpha, const float *A, const int* ldA, const float *B, const int* ldB, const float *beta, float *C, const int* ldC);
    void ssymm_(char* side, char* uplo, const int* m, const int* n, 
                const float *alpha, const float *A, const int* ldA, const float *B, const int* ldB, const float *beta, float *C, const int* ldC);
    void ssyrk_(char* uplo, char* trans, const int* n, const int* k, 
                const float *alpha, const float *A, const int* ldA, const float *beta, float *C, const int* ldC);
    void ssyr2k_(char* uplo, char* trans, const int* n, const int* k, 
                 const float *alpha, const float *A, const int* ldA, const float *B, const int* ldB, const float *beta, float *C, const int* ldC);
    void strmm_(char*, char*, char*, char*, const int*, const int*, const float *, const float *, const int*, float *, const int*);
    void strsm_(char*, char*, char*, char*, const int*, const int*, const float *, const float *, const int*, float *, const int*);

/* Double Precision */

    void dgemm_(char* transA, char* transB, const int* m, const int* n, const int* k, 
                const double *alpha, const double *A, const int* ldA, const double *B, const int* ldB, const double *beta, double *C, const int* ldC);
    void dsymm_(char* side, char* uplo, const int* m, const int* n, 
                const double *alpha, const double *A, const int* ldA, const double *B, const int* ldB, const double *beta, double *C, const int* ldC);
    void dsyrk_(char* uplo, char* trans, const int* n, const int* k, 
                const double *alpha, const double *A, const int* ldA, const double *beta, double *C, const int* ldC);
    void dsyr2k_(char* uplo, char* trans, const int* n, const int* k, 
                 const double *alpha, const double *A, const int* ldA, const double *B, const int* ldB, const double *beta, double *C, const int* ldC);
    void dtrmm_(char*, char*, char*, char*, const int*, const int*, const double *, const double *, const int*, double *, const int*);
    void dtrsm_(char*, char*, char*, char*, const int*, const int*, const double *, const double *, const int*, double *, const int*);

/* Single Complex Precision */

    void cgemm_(char*, char*, const int*, const int*, const int*, const float *, const float *, const int*, const float *, const int*, const float *, float *, const int*);
    void csymm_(char*, char*, const int*, const int*, const float *, const float *, const int*, const float *, const int*, const float *, float *, const int*);
    void chemm_(char*, char*, const int*, const int*, const float *, const float *, const int*, const float *, const int*, const float *, float *, const int*);
    void csyrk_(char*, char*, const int*, const int*, const float *, const float *, const int*, const float *, float *, const int*);
    void cherk_(char*, char*, const int*, const int*, const float *, const float *, const int*, const float *, float *, const int*);
    void csyr2k_(char*, char*, const int*, const int*, const float *, const float *, const int*, const float *, const int*, const float *, float *, const int*);
    void cher2k_(char*, char*, const int*, const int*, const float *, const float *, const int*, const float *, const int*, const float *, float *, const int*);
    void ctrmm_(char*, char*, char*, char*, const int*, const int*, const float *, const float *, const int*, float *, const int*);
    void ctrsm_(char*, char*, char*, char*, const int*, const int*, const float *, const float *, const int*, float *, const int*);

/* Double Complex Precision */

    void zgemm_(char*, char*, const int*, const int*, const int*, const double *, const double *, const int*, const double *, const int*, const double *, double *, const int*);
    void zsymm_(char*, char*, const int*, const int*, const double *, const double *, const int*, const double *, const int*, const double *, double *, const int*);
    void zhemm_(char*, char*, const int*, const int*, const double *, const double *, const int*, const double *, const int*, const double *, double *, const int*);
    void zsyrk_(char*, char*, const int*, const int*, const double *, const double *, const int*, const double *, double *, const int*);
    void zherk_(char*, char*, const int*, const int*, const double *, const double *, const int*, const double *, double *, const int*);
    void zsyr2k_(char*, char*, const int*, const int*, const double *, const double *, const int*, const double *, const int*, const double *, double *, const int*);
    void zher2k_(char*, char*, const int*, const int*, const double *, const double *, const int*, const double *, const int*, const double *, double *, const int*);
    void ztrmm_(char*, char*, char*, char*, const int*, const int*, const double *, const double *, const int*, double *, const int*);
    void ztrsm_(char*, char*, char*, char*, const int*, const int*, const double *, const double *, const int*, double *, const int*);

}


namespace PLearn {

// Overload for float and double and passing integers and real scalars by value
// (we did not yet write all overloads, only those we currently use, please feel free to add other ones)

inline float blas_dot(int n, const float* x, int incx, const float* y, int incy)
{ return sdot_(&n, x, &incx, y, &incy); }

inline double blas_dot(int n, const double* x, int incx, const double* y, int incy)
{ return ddot_(&n, x, &incx, y, &incy); }

inline void blas_scal(int n, float alpha, float *x, int incx)
{ sscal_( &n, &alpha, x, &incx); }

inline void blas_scal(int n, double alpha, double *x, int incx)
{ dscal_( &n, &alpha, x, &incx); }

inline void blas_axpy(int n, float alpha, const float *x, int incx, float *y, int incy)
{ saxpy_(&n, &alpha, x, &incx, y, &incy); }

inline void blas_axpy(int n, double alpha, const double *x, int incx, double *y, int incy)
{ daxpy_(&n, &alpha, x, &incx, y, &incy); }

inline void blas_ger(int m, int n, float alpha, const float *x, int incx, const float *y, int incy, float *A, int ldA)
{ sger_(&m, &n, &alpha, x, &incx, y, &incy, A, &ldA); }

inline void blas_ger(int m, int n, double alpha, const double *x, int incx, const double *y, int incy, double *A, int ldA)
{ dger_(&m, &n, &alpha, x, &incx, y, &incy, A, &ldA); }

inline void blas_syr(char uplo, int n, float alpha, const float *x, int incx, float *A, int ldA)
{ ssyr_(&uplo, &n, &alpha, x, &incx, A, &ldA); }

inline void blas_syr(char uplo, int n, double alpha, const double *x, int incx, double *A, int ldA)
{ dsyr_(&uplo, &n, &alpha, x, &incx, A, &ldA); }

inline void blas_syr2(char uplo, int n, float alpha, const float *x, int incx, const float *y, int incy,  float *A, int ldA)
{ ssyr2_(&uplo, &n, &alpha, x, &incx, y, &incy,  A, &ldA); }

inline void blas_syr2(char uplo, int n, double alpha, const double *x, int incx, const double *y, int incy,  double *A, int ldA)
{ dsyr2_(&uplo, &n, &alpha, x, &incx, y, &incy,  A, &ldA); }

inline void blas_gemv(char trans, int m, int n, float alpha, const float *A, int ldA, const float *x, int incx, float beta, float *y, int incy)
{ sgemv_(&trans, &m, &n, &alpha, A, &ldA, x, &incx, &beta, y, &incy); }

inline void blas_gemv(char trans, int m, int n, double alpha, const double *A, int ldA, const double *x, int incx, double beta, double *y, int incy)
{ dgemv_(&trans, &m, &n, &alpha, A, &ldA, x, &incx, &beta, y, &incy); }

inline void blas_symv(char uplo, int n, float alpha, const float *A, int ldA, const float *x,  int incx, float beta, float *y, int incy)
{ ssymv_(&uplo, &n, &alpha, A, &ldA, x, &incx, &beta, y, &incy); }

inline void blas_symv(char uplo, int n, double alpha, const double *A, int ldA, const double *x,  int incx, double beta, double *y, int incy)
{ dsymv_(&uplo, &n, &alpha, A, &ldA, x, &incx, &beta, y, &incy); }

inline void blas_gemm(char transA, char transB, int m, int n, int k, 
                      float alpha, const float *A, int ldA, const float *B, int ldB, float beta, float *C, int ldC)
{ 
    // fprintf(stderr, "??? sgemm last param ldC = %d ( %d ) \n", ldC, C);
    sgemm_(&transA, &transB, &m, &n, &k, &alpha, A, &ldA, B, &ldB, &beta, C, &ldC); }

inline void blas_gemm(char transA, char transB, int m, int n, int k, 
                      double alpha, const double *A, int ldA, const double *B, int ldB, double beta, double *C, int ldC)
{ dgemm_(&transA, &transB, &m, &n, &k, &alpha, A, &ldA, B, &ldB, &beta, C, &ldC); }

inline void blas_symm(char side, char uplo, int m, int n, 
                      float alpha, const float *A, int ldA, const float *B, int ldB, float beta, float *C, int ldC)
{ ssymm_(&side, &uplo, &m, &n, &alpha, A, &ldA, B, &ldB, &beta, C, &ldC); }

inline void blas_symm(char side, char uplo, int m, int n, 
                      double alpha, const double *A, int ldA, const double *B, int ldB, double beta, double *C, int ldC)
{ dsymm_(&side, &uplo, &m, &n, &alpha, A, &ldA, B, &ldB, &beta, C, &ldC); }

inline void blas_syrk(char uplo, char trans, int n, int k, 
                      float alpha, const float *A, int ldA, float beta, float *C, int ldC)
{ ssyrk_(&uplo, &trans, &n, &k, &alpha, A, &ldA, &beta, C, &ldC); }

inline void blas_syrk(char uplo, char trans, int n, int k, 
                      double alpha, const double *A, int ldA, double beta, double *C, int ldC)
{ dsyrk_(&uplo, &trans, &n, &k, &alpha, A, &ldA, &beta, C, &ldC); }

inline void blas_syr2k(char uplo, char trans, int n, int k, 
                        float alpha, const float *A, int ldA, const float *B, int ldB, float beta, float *C, int ldC)
{ ssyr2k_(&uplo, &trans, &n, &k, &alpha, A, &ldA, B, &ldB, &beta, C, &ldC); }

inline void blas_syr2k(char uplo, char trans, int n, int k, 
                        double alpha, const double *A, int ldA, const double *B, int ldB, double beta, double *C, int ldC)
{ dsyr2k_(&uplo, &trans, &n, &k, &alpha, A, &ldA, B, &ldB, &beta, C, &ldC); }


} // end of namespace PLearn

#endif /*  blas_proto_overload_INC */


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
