// -*- C++ -*-

// CudaSparseTargetFactoredLinearOutputLayer.h
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

#ifndef CudaSparseTargetFactoredLinearOutputLayer_INC
#define CudaSparseTargetFactoredLinearOutputLayer_INC

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>

#include <cstring>
#include <string>

// #define real float
#include "pl_math.h"

#include <cublas_linalg.h>
#include <cublas_k_sparse.h>

#include <CudaSparseTargetLinearOutputLayer.h>

using std::string;
using std::strcmp;
using namespace PLearn;

// TODO: check we get same result when use_lower is false (and compare speed to see which alternative is faster).

// TODO: currently grad_H is set to thegradient. Maybe to more easily allow other influences we could accumulate into it 
// (to do: make it an option of the call). In that case, the user needs to clear it first before starting accumulating gradients in it.

#define real float

namespace PLearn {


// Performs rank one update to U^-T (inverse of U transposed) that corresponds to rank 1 update to U <- U + alpha u v^T
// using Sherman-Morrison formula.
void blas_rank_1_update_UinvT(const CublasMat<real>& UinvT, real alpha, const CublasVec<real>& u, const CublasVec<real>& v) 
{
    // d-dimensional vectors for intermediate computations
    static CublasVec<real> u_tilde;  
    static CublasVec<real> v_tilde;  

    int d = UinvT.nrows();
    u_tilde.resize(d);
    v_tilde.resize(d);

    // u_tilde = UinvT^T u
    blas_gemv('T', 1, UinvT, u, 0, u_tilde);
    // v_tilde = UinvT v
    blas_gemv('N', 1, UinvT, v, 0, v_tilde);
    real s = blas_dot(v, u_tilde);
    real alpha_tilde = -alpha/(1+alpha*s);
    // UinvT = UinvT + alpha_tilde v_tilde u_tilde^T
    blas_ger(alpha_tilde, v_tilde, u_tilde, UinvT); 
}

// Performs rank k update to U^-T (inverse of U transposed) that corresponds to rank k update to U <- U + alpha A B^T
// where U is a d x d matrix,  A and B are d x k matrices and alpha is a scalar. 
// This is done by calling k times rank-one updates
void blas_rank_update_UinvT(const CublasMat<real>& UinvT, real alpha, const CublasMat<real>& A, const CublasMat<real>& B)  
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

void rankm_update_U_and_UinvT_iter_v1(real alpha, const CublasMat<real>& A, const CublasMat<real>& B, const CublasMat<real>& U, const CublasMat<real>& UinvT)  
  
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
void rankm_update_U_and_UinvT_iter_v2(real alpha, const CublasMat<real>& A, const CublasMat<real>& B, const CublasMat<real>& U, const CublasMat<real>& UinvT)
{
    assert(A.nrows()==B.nrows() && A.ncols()==B.ncols());

    // d-dimensional vectors for intermediate results
    static CublasVec<real> u;
    static CublasVec<real> v;

    int d = U.nrows();
    u.resize(d);
    v.resize(d);

    // 7) U <- U + alpha A B^T
    blas_gemm_NT(alpha, A, B, 1, U);

    // 8) Corresponding update to UinvT

    for (int k=0; k<B.ncols(); k++)
    {
        CublasVec<real> Bk = B.column(k);
        blas_gemv('N', 1, UinvT, Bk, 0, u);

        if (k==0)
        {
            real Bk_v = blas_dot(Bk,Bk);          
            real scale = -alpha / (1+alpha*Bk_v);
            blas_ger(scale, u, Bk, UinvT);
        }
        else
        {
            CublasVec<real> Ak = A.column(k);          
            blas_gemv('T', 1, UinvT, Ak, 0, v);
            real Bk_v = blas_dot(Bk, v);
            real scale = -alpha / (1+alpha*Bk_v);
            blas_ger(scale, u, v, UinvT);
        }
    }
}


class CudaSparseTargetFactoredLinearOutputLayer: public CudaSparseTargetLinearOutputLayer
{

protected:
    CudaStreamArray cuda_streams;
    CudaEventArray cuda_events;

    // Constant indexes in cuda_events, representing events to be used
    static const int ISREADY_UinvT_B = 1;
    static const int ISREADY_Z_hat = 2;
    static const int ISREADY_grad_H = 3;

public:

    int m; // minibatch size

    bool use_Qtilde; // if true we'll use the algo that updates Qtilde if false, we'll update and use Q instead
    bool use_lower; // if true, will use only lower triangular part of Q or Qtilde
    int invup_mode; // determines how we will update the inverse of U UinvT
    // invup_mode has the following meaning 
    //  1: don't kep an up-to-date UinvT, but solve a linear system each time
    //  2: iterate rank-1 updates; 
    //  3: iterate rank-one updates with special handling of first update (saves one matrix vector product)
    //  4: use Woodbury identity
    //  5: recompute full inverse
    //  6: use Woodbury identity, streamed version
    // If passed as negative values (e.g. -6), then matrix inverses and linear system solves will be done on gpu (cublas) rather than cpu (lapack)

    // Parameter initialization specification
    real udiag;
    real iVlow;
    real iVhigh; 
    int iVk;
    
    // Parameters for controling numerical stability 
    int nupdates; // Number of updates since initialization

    int renormalize_period;  // every how many updates to renormalize
    int unfactorize_period;  // every how many updates to unfactorize
    int stabilize_period;  // every how many updates to singular_stabilize

    // Parameters U and V and bookkeeping matrices (all are updated by the call)
    CublasMat<real> U;         // d x d
    CublasMat<real> UinvT; // d x d This is up-to-date U^-T
    real det_U;         // scalar: determinant of U
    CublasMat<real> VT;    // d x D matrix VT = V^T
    CublasMat<real> Qtilde;     // d x d This is up-to-date V^T V
    CublasMat<real> Q;       // d x d this is up-to-date W^T W = U^T V^T V U
    
    // Allocated temporary work matrices
    CublasMat<real> M;        // m x m  
    CublasMat<real> TmpWork_d;  // (d, 4m)

public:

    // Allocates, computes and returns WT = U^T V^T
    virtual CublasMat<real> get_WT() const 
    {
        CublasMat<real> WT(d,D);
        blas_gemm_TN(1, U, VT, 0, WT);
        return WT;
    }

    //! Computes WT = U^T VT
    void compute_WT(const CublasMat<real>& WT) const
    {
        blas_gemm_TN(1, U, VT, 0, WT);
    }

    virtual void operator<<(const CudaSparseTargetLinearOutputLayer& other)
    {
        const CudaSparseTargetFactoredLinearOutputLayer* other_factored = dynamic_cast<const CudaSparseTargetFactoredLinearOutputLayer*>(&other);
        if (other_factored != 0)
        {
            U << other_factored->U;
            UinvT << other_factored->UinvT;
            det_U = other_factored->det_U;
            VT << other_factored->VT;
            if (use_Qtilde) // this algo uses Qtilde
            {
                if (other_factored->use_Qtilde) // other algo also based on Qtilde
                    Qtilde << other_factored->Qtilde;
                else // other algo based on Q, we need to recompute Qtilde = V^T V from scratch
                    blas_syrk('L', 1, VT, 0, Qtilde);
                // symmetrize it
                copy_lower_to_upper(Qtilde);
            }
            else // this algo uses Q
            {
                if (!other_factored->use_Qtilde) // other algo also based on Q
                {
                    Q << other_factored->Q;
                    // symmetrize it
                    copy_lower_to_upper(Q);
                }
                else // other algo based on Qtilde
                {   // we'll use the other algo's Qtilde to infer Q = U^T Qtilde U 
                    CublasMat<real> Qtilde_U(d,d); // temporary
                    blas_symm('L', 1, other_factored->Qtilde, other_factored->U, 0, Qtilde_U);
                    blas_gemm_TN(1, other_factored->U, Qtilde_U, 0, Q);

                    // Note: this more efficient version could be used when U = udiag I
                    // Q << other_factored->Qtilde;
                    // Q *= other_factored->udiag*other_factored->udiag;                  
                }
            }             
        }
        else // default copying from other algo
        {    // we set this VT to the others' WT and U to the identity
            VT << other.get_WT();
            U.clear();
            add_scalar_to_diagonal(1, U);  
            if (use_Qtilde)
                blas_syrk('L', 1, VT, 0, Qtilde);
            else // this algo uses Q: with U=I we have Q = Qtilde = V^T V
                blas_syrk('L', 1, VT, 0, Q);
        }
    }

    CudaSparseTargetFactoredLinearOutputLayer()
        :CudaSparseTargetLinearOutputLayer(),
         m(-1),
         use_Qtilde(true), use_lower(true), invup_mode(6),
         udiag(1), iVlow(-1), iVhigh(1), iVk(1),
         nupdates(0), renormalize_period(0), unfactorize_period(0), stabilize_period(0),
         det_U(-1)
    {
        cuda_streams = CudaStreamArray(8);
        cuda_events = CudaEventArray(2*cuda_streams.length());        
    }
    


    void sanity_check_dimensions() const
    {
        // Parameter and bookkeeping matrices
        assert(m>0);
        assert( U.nrows() == d && U.ncols() == d );
        assert( UinvT.nrows() == d && UinvT.ncols() == d );
        assert( VT.nrows() == d && VT.ncols() == D );
        assert( (use_Qtilde && Qtilde.nrows() == d && Qtilde.ncols() == d) || (!use_Qtilde && Q.nrows()==d && Q.ncols()==d) );

        // Work matrices
        assert( M.nrows() == m && M.ncols() == m );
        assert( TmpWork_d.nrows() == d && TmpWork_d.ncols() >= 4*m );
    }
    
    void allocate_params()
    {
        U = CublasMat<real>(d,d);
        UinvT = CublasMat<real>(d,d);
        VT = CublasMat<real>(d,D);
        if (use_Qtilde)
            Qtilde = CublasMat<real>(d,d);  
        else
            Q = CublasMat<real>(d,d);  
    }
    
    void resize_work_storage() 
    {
        assert(m>0);
        M.resize(m,m);
        TmpWork_d.resize(d,4*m);
    }

//! Initializes parameters U and VT based on specified properties
//! udiag, iVlow, iVhigh, iVk
//! VT will be initialized as a d x D matrix with each column having iVk non-zero elements between iVlow and iVhigh
//! U will be initialized as a d x d diagonal matrix with value udiag on the diagonal;
    void initialize_params()
    {
        // Initialize U and UinvT as d x d diagonal matrices.
        U.clear();
        add_scalar_to_diagonal(udiag, U);  // U has udiag as its diagonal values
        UinvT.clear();
        add_scalar_to_diagonal(1/udiag, UinvT); // UinvT has 1/udiag as its diagonal values
        // determinant of U
        det_U = (real) pow(udiag,d);
        
        // For now make Qt into an alias for Qtilde, unless there's no Qtilde in which case we make it an alias for Q
        CublasMat<real> Qt;
        if (Qtilde.isNotEmpty())            
            Qt = Qtilde;  // Make Qt an alias for Qtilde
        else
            Qt = Q;  // Make Qt an alias for Q
        
        // Initialize VT (d x D colum-major matrix) and associated Qt = V^T V

        // number of non-zero elems to generate in each colum of VT is iVk not related with the K-sparsity of the target) 

        if (iVk<=0 || iVk>=d) // random initialize full VT 
        {
            // We'll initialize a CPU version and copy it to the gpu matrix
            BMat<real> VT_cpu(VT.nrows(), VT.ncols());
            fill_random_uniform_continuous(VT_cpu, iVlow, iVhigh);
            VT_cpu >> VT;
            blas_syrk('L', 1, VT, 0, Qt);  // only lower part of Qt is computed.
            copy_lower_to_upper(Qt);
        }

        else if (iVk==1)  // Let's initialise each column of VT as sparse with only one non-zero element 
        {
            // Note that this results in a diagonal Qt
            VT.fill(0);
            Qt.fill(0);
            double range = iVhigh-iVlow;
            for (int col=0; col<D; col++)
            {
                int i = rand()%d;
                real val = (real)(iVlow + range*(((double)rand())/RAND_MAX));
                VT.set(i, col, val);
                Qt.set(i,i, Qt(i,i)+val*val);
            }
        }
        else // Let's initialise each column of VT as sparse with number of non-zero elements beign iVk  
        {
            VT.fill(0);
            Qt.fill(0);

            BVec<int> indexes(iVk);
            BVec<real> values(iVk);
      
            for (int col=0; col<D; col++)
            {
                fill_random_uniform_discrete_noduplicate(indexes, 0, d-1);
                fill_random_uniform_continuous(values, iVlow, iVhigh);          
                int ii, jj;

                for(jj=0; jj<iVk; jj++) 
                {
                    int j = indexes[jj];
                    real val_j = values[jj];
                    VT.set(j,col,val_j);

                    for (ii=0; ii<iVk; ii++)
                    {
                        int i = indexes[ii];
                        real val_i = values[ii];
                        Qt.set(i,j, Qt(i,j)+val_i*val_j);
                    }
                }
            }

        }

        // At this stage we have initialized VT and correpsonding Qt = V^T V
        // Note that Q = W^T W = U^T V^T V U = U^T Qt U
        // With a diagonal U = udiag I this yields Q = (udiag*udiag) Qt

        // Do we actually have a Q to compute?
        if (Q.isNotEmpty() ) // Yes
        {
            if ( Qtilde.isNotEmpty() ) // our Qt was actually an alias for Qtilde
                Q << Qtilde; // so copy it over to Q
            // otherwise Qt was already set up as an alias to Q so no copying is needed

            // now rescale by square of udiag
            Q *= (udiag*udiag);
        }

        /*
        // Sanity checking: verifying hand-computed Qtilde in sparse case 
        printf("\nQtilde hand-computed on the go as we initialize sparse VT:\n");
        print(Qtilde);
        printf("\n Lower part of Qtilde recomputed with blas after the fact: \n");
        Qtilde.fill(0);
        blas_syrk('L', 1, VT, 0, Qtilde);  // only lower part of Qtilde is used.
        print(Qtilde);     

        // Print UinvT alsoto see if it's the identity 
        printf("UinvT:\n");
        print(UinvT);
        */

    } 

    void build()
    {
        if (U.isEmpty())
        {
            allocate_params();
            initialize_params();
        }
        if (TmpWork_d.isEmpty())
        {
            resize_work_storage(); 
        }

        sanity_check_dimensions();
    }

    CudaSparseTargetFactoredLinearOutputLayer(int d_, int D_, int K_, int m_,
                                         bool use_Qtilde_ = true,
                                         real udiag_=1, real iVlow_=-1, real iVhigh_=1, int iVk_=1)
        :CudaSparseTargetLinearOutputLayer(d_, D_, K_), 
         m(m_),
         use_Qtilde(use_Qtilde_), use_lower(true), invup_mode(6),
         udiag(udiag_), iVlow(iVlow_), iVhigh(iVhigh_), iVk(iVk_),
         nupdates(0), renormalize_period(0), unfactorize_period(0), stabilize_period(0),
         det_U(-1)
    {

        cuda_streams = CudaStreamArray(8);
        cuda_events = CudaEventArray(2*cuda_streams.length());        

        build();
    }

//! Applys a gradient update corresponding to a weight decay on W of strength lambda, using learning rate eta
//! This is done by rescaling U and UinvT
    virtual void apply_weight_decay_update(real lambda, real eta)
    {
        real s = 1-eta*lambda;
        U *= s;
        UinvT *= (1/s);    
    }

    
//! This call multiplies the columns of V (the rows of VT) by the given scales (a d-dimensional vector)
//! It correspondingly downscales U so that W=VU is unchanged, and updates UinvT and Qtilde to reflect these changes
//! Beware that scales is changed by the call (set to its inverse)
void scale_VT_rows(const CublasVec<real>& scales)
{
    // printf("Scaling rows of VT by "); print(scales); printf("/n");
    // VT = diag(diag(U)/udiag) VT 
    scale_rows(VT, scales);

    // Correspondingly update Qtilde (note that Q is unchanged, since we will correpsondingly downscale U)
    if (Qtilde.isNotEmpty())
    {
        scale_rows(Qtilde, scales);
        scale_columns(Qtilde, scales);
        /*
       int d = U.nrows();
       for (int i=0; i<d; i++)
            for(int j=0; j<d; j++)
                Qtilde(i,j) = Qtilde(i,j)*(scales[i]*scales[j]);
        */
    }

    // UinvT = diag(diag(UinvT)/udiag) UinvT 
    scale_rows(UinvT, scales);

    // nowinvert scales so that scales = udiag/diag(U)
    // printf("udiag = %e\n", udiag);
    invert_elements(scales);
    det_U *= product(scales);
    
    /*
    for( int i=0; i<d; i++)
    {
        scales[i] = 1/scales[i];
        det_U *= scales[i];
    }
    */

    // U = diag( udiag / diag(U) ) U
    // ensures that thediagonal elements of U become udiag
    scale_rows(U, scales);

}

//! Checks consistency of U and UinvT and of Qtilde and VT
    virtual void consistency_check() const
    {
        CublasMat<real> C(d,d); // allocate temporary matrix
        // Compute C = U UinvT^T - I
        blas_gemm_NT(1, U, UinvT, 0, C);
        add_scalar_to_diagonal(-1, C); 
        printf( "Conxxxxsistency checking U and UinvT: max abs (U UinvT^T - I) = %e\n", max_abs(C) );
        
        BMat<real> svd_U;
        BMat<real> svd_VT;
        BVec<real> svd_s;
        BMat<real> U_cpu(U.nrows(), U.ncols());
        U_cpu << U;
        lapackSVD(U_cpu, svd_U, svd_s, svd_VT, 'A');
        printf("Singular value spectrum of U: "); print(svd_s);
        printf("Product of singular values: %g\n", product(svd_s));
        printf("Condition number: %g / %g = %g \n", svd_s[0], svd_s[d-1], svd_s[0]/svd_s[d-1]);
        
        // Compute C = V^T V 
        blas_syrk('L', 1, VT, 0, C);
        // make sure it's symmetric
        copy_lower_to_upper(C);

        if (Qtilde.isNotEmpty()) // compare C to Qtilde
        {
            if (use_lower) // only th lower part of Qtilde has been updated
                copy_lower_to_upper(Qtilde);  // symmetrize it for our comparisons
            printf( "Consistency checking V and Qtilde: max abs (V^T V - Qtilde) = %e\n", max_abs_diff(C,Qtilde) );
        }

        if (Q.isNotEmpty()) // Compare U^T C U = U^T V^T V U = W^T W to Q
        {
            if (use_lower) // only the lower part of Q has been updated
                copy_lower_to_upper(Q);  // symmetrize it for our comparisons

            CublasMat<real> CU(d,d); // allocate temporary matrix
            blas_gemm_NN(1, C, U, 0, CU);
            blas_gemm_TN(1, U, CU, 0, C);
            printf( "Consistency checking Q: max abs (U^T (V^T V) U - Q) = %e\n", max_abs_diff(C,Q) );
            CublasMat<real> WT = get_WT();
            blas_gemm_NT(1, WT, WT, 0, C);
            printf( "Consistency checking Q: max abs ( WT W - Q) = ( (U^T VT) (U^T VT)^T - Q ) = %e\n", max_abs_diff(C,Q) );
        }
    }
    
//! This may be called periodically for numerical stability reasons.
//! It brings back the diagonal of U to being udiag (but U will not be made diaognal, only have its rows rescaled)
//! It correspondingly rescales the columns of V (the rows of VT) the rows of UinvT and updates Qtilde
//! tmp_scale is a d-dimensional temporary vector that will be used by the call
    void renormalize_U(real udiag)
    {
        CublasVec<real> tmp_scales = TmpWork_d.column(0);
        tmp_scales << U.diagonal();
        tmp_scales *= 1/udiag;
        scale_VT_rows(tmp_scales);
    }

    void renormalize_VT(real VT_norm_target)
    {
        CublasVec<real> tmp_scales = TmpWork_d.column(0);
        tmp_scales << Qtilde.diagonal();
        sqrt_elements(tmp_scales);
        invert_elements(tmp_scales);
        tmp_scales *= VT_norm_target;
        scale_VT_rows(tmp_scales);
    }

//! unfactorizing: getting back to vanilla state where U = udiag I
//! Since this call essentially computed W=VU it is O(d^2 D)
//! This call currently allocates a temporary work matrix new_WT of same dimensions as VT (i.e. d x D )
//! which is freed upon exiting
    void unfactorize(real udiag=1)
    {
        CublasMat<real> new_WT(d,D); 

        // VT = U^T VT
        blas_gemm_TN(1/udiag, U, VT, 0, new_WT);
        VT << new_WT;

        if (Qtilde.isNotEmpty())
        {
            // Qtilde = V^T T        
            blas_syrk('L', 1, VT, 0, Qtilde);
            copy_lower_to_upper(Qtilde);
        }

        // U = udiag * I  and UinvT = (1/udiag) * I
        U.clear();
        add_scalar_to_diagonal(udiag, U);  // U has udiag as its diagonal values
        UinvT.clear();
        add_scalar_to_diagonal(1/udiag, UinvT); // UinvT has 1/udiag as its diagonal values
        // determinant of U
        det_U = (real) pow(udiag, U.nrows());
    }

    //! Will bring back the given singular value of U associated with the gven singular vector
    //! back to being equal to target_singval
    void singular_stabilize_U(real singval, const CublasVec<real>& singvec, real target_singval)
    {
        // Bringing back the largest and smallest singular values to value target_singval
        real s = singval;
        const CublasVec<real>& u = singvec;

        real alpha = (target_singval-s)/s;

        // Let us define C = (I + alpha u u^T)
        // Note that CU should correct the smallest singular value of U bringing it back to value target_singval
        // 
        // Now we want to left-multiply U by C and right-multiply V by C^-1
        //   so that we haven't changed anything to W=VU        
        //   : VU = (V C^-1) (C U)
        // So we will actually update U <- C U  and V <- V C^-1
        // From Sherman-Morrison formula we have
        // C^-1 = (I + (alpha u) u^T)^-1 
        //      = I^-1 - ( I^-1 (alpha u) u^T I^-1 )/(1+u^T I^-1 (alpha u))
        //      = I - ( alpha u u^T ) / (1 + alpha u^T u )
        //      = I - ( alpha / (1+ alpha u^T u) ) u u^T
        // where we have here u^T u = 1
        // So C^-1 = I + beta u u^T
        //    with  beta = - alpha / (1+ alpha) 

        // The updates to U and VT thus read
        // U <- C U
        //   <- (I + alpha u u^T) U
        //   <- U + alpha u (U^T u)^T
        // And
        // V <- V C^-1
        //   <- V ( I + beta u u^T )
        //   <- V + beta (V u) u^-T
        // or equivalently 
        // VT <- VT + beta u (V u)^T

        // We also need to update
        // U^-1 <- (C U)^-1 = U^-1 C^-1 = U^-1 ( I + beta u u^T ) 
        //      <- U^-1 + beta (U^-1 u) u^T
        // Or equivalently
        // U^-T <- U^-T + beta u (U^-1 u)^T

        real beta = -alpha / (1 + alpha);

        // Do U <-  U + alpha u (U^T u)^T
        CublasVec<real> UT_u(d);
        blas_gemv('T', 1, U, u, 0, UT_u);
        blas_ger(alpha, u, UT_u, U);

        // Do UinvT <- UinvT + beta u (U^-1 u)^T
        CublasVec<real> Uinv_u(d);
        blas_gemv('T', 1, UinvT, u, 0, Uinv_u);
        blas_ger(beta, u, Uinv_u, UinvT);

        // Do VT <- VT + beta u (V u)^T
        CublasVec<real> Vu(D);
        blas_gemv('T', 1, VT, u, 0, Vu);
        blas_ger(beta, u, Vu, VT);
        
        // In principle we don't need to update Q since W=VU is unchanged.

        // Now if we maintain a Qtilde=V^T V we should update Qtilde.
        // We updated VT <- VT + beta u (V u)^T
        // This yields Qtilde <- (VT + beta u (V u)^T) (V + beta (V u) u^T )
        //                    <- VT V + beta u (V u)^T V + beta VT (V u) u^T + beta^2 u (V u)^T (V u) u^T
        //                    <- Qtilde + beta u u^T V^T V + beta V^T V u u^T + beta^2 u u^T V^T V u u^T
        //                    <- Qtilde + beta u (V^T V u)^T + beta (V^T V u) u^T + beta^2 u (u^T Qtilde u) u^T
        //                    <- Qtilde + beta u (Qtilde u)^T + beta (Qtilde u) u^T + (beta^2 (u^T (Qtilde u))) u u^T
        if (Qtilde.isNotEmpty())
        {
            CublasVec<real> Qtilde_u(d);            
            if (use_lower)  // use and update only lower part of Qtilde
            {
                blas_symv('L', 1, Qtilde, u, 0, Qtilde_u);
                blas_syr2('L', beta, u, Qtilde_u, Qtilde);
                blas_syr('L', beta*beta*blas_dot(u, Qtilde_u), u, Qtilde);
            }
            else
            {
                blas_gemv('N', 1, Qtilde, u, 0, Qtilde_u);
                blas_ger(beta, u, Qtilde_u, Qtilde);
                blas_ger(beta, Qtilde_u, u, Qtilde);
                blas_ger(beta*beta*blas_dot(u, Qtilde_u), u, u, Qtilde);                
            }
        }

    }
    
    // This call does a SVD of U followed by an update to V that is O(2 Dd)
    // It corrects the smallest singular value of U so that its value becomes udiag
    // (which for this call you can specify to have a different value than the member variable target_singval)
    void singular_stabilize_OLDER(real target_singval, int n_power_iterations=100) 
    {
        cudaDeviceSynchronize();
        cublas.setStream(cublas.streams[0]);

        // Find leading left singular vector of U (assosciated to largest singular value) by power iteration
        CublasVec<real> leading_left_singvec(U.nrows());
        CublasVec<real> leading_right_singvec(U.ncols());
        leading_left_singvec.fill((real)1);
        real largest_singval = power_iteration_for_singular_vect(U, leading_left_singvec, leading_right_singvec, n_power_iterations);

        // Find trailing left singular vector of U (associated to smallest singular value) 
        // It is the same as the leading left singular vector of UinvT, which we will compute by power iteration 

        if(std::abs(invup_mode)!=5) // Let us periodically recompute UinvT because iterated sherman morrison or woodbury lead to divergence
        {
            transpose(U, UinvT);
            //UinvT << U;
            //transpose_squaremat_inplace(UinvT);
            invertMatrix(UinvT, (invup_mode<0));
        }
        CublasVec<real> trailing_left_singvec(UinvT.nrows());
        CublasVec<real> trailing_right_singvec(UinvT.ncols());
        trailing_left_singvec.fill((real)1);
        real smallest_singval = 1/power_iteration_for_singular_vect(UinvT, trailing_left_singvec, trailing_right_singvec, n_power_iterations);

        if (debug_print>=1)
            printf("@@@ largest singval / smallest singval =  condition number: %e / %e = %e\n", largest_singval, smallest_singval, largest_singval/smallest_singval);

        // Bringing back the largest and smallest singular values to value target_singval
        if (largest_singval>2*target_singval)
            singular_stabilize_U(largest_singval, leading_left_singvec, target_singval);
        if (smallest_singval<0.5*target_singval)
            singular_stabilize_U(smallest_singval, trailing_left_singvec, target_singval);


        cudaDeviceSynchronize();

        /*
          // Same but finding leading and 

        int d = U.nrows();
        CublasMat<real> C(d,d); // allocate temporary matrix
        
        // First compute svd
        CublasMat<real> svd_U;
        CublasMat<real> svd_VT;
        CublasVec<real> svd_s;
        C << U;
        lapackSVD(C, svd_U, svd_s, svd_VT, 'A');

        // Compare leading and trailing left singular vectors obtained through SVD with the ones found through power iteration
        if (debug_print>=1)
        {
            printf("@@@ Max abs diff for leading singular vector: %e\n", max_abs_diff(leading_left_singvec, svd_U.column(d-1)));
            printf("@@@ Max abs diff for trailing singular vector: %e\n", max_abs_diff(trailing_left_singvec, svd_U.column(d-1)));
        }

        // Bringing back the largest and smallest singular values to value target_singval
        singular_stabilize_U(svd_s[0], svd_U.column(0), target_singval);
        singular_stabilize_U(svd_s[d-1], svd_U.column(d-1), target_singval);

        // update the determinant of U
        // we corrected the largest and smallest sigular values to now be equal to target_singval
        det_U = product(svd_s.subVec(1,d-1))*target_singval*target_singval;

        // For DEBUG we can print the previous singular value spectrum of U and then call consistency_check()
        if (debug_print>=1)
        {
            printf("@@@ In singular_stabilize: sprectum before: "); print(svd_s);
            printf("@@@ condition number before: %g / %g = %g \n", svd_s[0], svd_s[d-1], svd_s[0]/svd_s[d-1]);
        }

        */

        // printf("@@@ And consistency_check after: \n");
        // consistency_check();
    }


    void print_svd_spectrum_of_U()
    {
        BMat<real> C(d,d); // allocate temporary matrix
        
        // First compute svd
        BMat<real> svd_U;
        BMat<real> svd_VT;
        BVec<real> svd_s;
        C << U;
        lapackSVD(C, svd_U, svd_s, svd_VT, 'A');
        printf("\n@@@+++ svd spectrum: ");
        print(svd_s);
    }

    // This call does a SVD of U followed by an update to V that is O(2 Dd)
    // It corrects the singular value of U that re too far out of range so that they are put back to being target_singval
    void singular_stabilize_using_svd(real target_singval) 
    {
        cudaDeviceSynchronize();
        cublas.setStream(cublas.streams[0]);

        int d = U.nrows();
        BMat<real> C(d,d); // allocate temporary matrix
        
        // First compute svd
        BMat<real> svd_U;
        BMat<real> svd_VT;
        BVec<real> svd_s;
        C << U;
        lapackSVD(C, svd_U, svd_s, svd_VT, 'A');

        if (debug_print>=2)
        {
            printf("\n@@@ svd spectrum of U : ");
            print(svd_s);
            printf("\n@@@ largest singval / smallest singval =  condition number: %e / %e = %e\n", svd_s[0], svd_s[d-1], svd_s[0]/svd_s[d-1]);
        }

        // count how many singvals are outside of range [0.5, 2]
        int n_outside_range = 0;
        for(int k=0; k<d; k++)
        {
            if (svd_s[k]<0.5*target_singval || svd_s[k]>2*target_singval)
                n_outside_range++;
        }

        if (debug_print>=2)
            printf("@@@ svd spectrum: %d singular values (out of %d) are outside range [0.5,1] \n", n_outside_range, d);

        if (n_outside_range*3 > d)  // if more than 1/3 are outside the range, we call unfactorize
        {
            printf("@@@ calling unfactorize (because too many singvals are outside accepted range)\n");
            unfactorize(target_singval);
        }
        else  // repeatedly call singular_stabilize_U
        {
            CublasVec<real> svd_uk(d);

            for(int k=d-1; k>=0; k--)
            {
                svd_uk << svd_U.column(k);
                if (svd_s[k]<0.5*target_singval || svd_s[k]>2*target_singval)
                {
                    singular_stabilize_U(svd_s[k], svd_uk, target_singval);
                    if (debug_print>=2)
                        print_svd_spectrum_of_U();
                }
            }
        }

        // cublas.streams.synchronize();
        cudaDeviceSynchronize();
    }

    // same as above, but leading and trailing singular values and vectors are computed using power iteration instead of SVD
    void singular_stabilize_using_power_iteration(real target_singval, int n_power_iterations) 
    {
        cudaDeviceSynchronize();
        // cublas.setStream(cublas.streams[0]);
        cublas.setStream(0);

        // Find leading and trailing left singular vector of U (assosciated to largest singular value) by power iteration
        CublasVec<real> leading_left_singvec(U.nrows());
        CublasVec<real> leading_right_singvec(U.ncols());
        CublasVec<real> trailing_left_singvec(UinvT.nrows());
        CublasVec<real> trailing_right_singvec(UinvT.ncols());

        // Find trailing left singular vector of U (associated to smallest singular value) 
        // It is the same as the leading left singular vector of UinvT, which we will compute by power iteration 
        if(std::abs(invup_mode)!=5) // Let us periodically recompute UinvT because iterated sherman morrison or woodbury lead to divergence
        {
            // UinvT << U;
            // transpose_squaremat_inplace(UinvT);
            transpose(U, UinvT);
            invertMatrix(UinvT, (invup_mode<0));
        }

        real largest_singval = power_iteration_for_singular_vect(U, leading_left_singvec, leading_right_singvec, n_power_iterations);
        real smallest_singval = 1/power_iteration_for_singular_vect(UinvT, trailing_left_singvec, trailing_right_singvec, n_power_iterations);
        if (debug_print>=2)
            printf("@@@ largest singval / smallest singval =  condition number: %e / %e = %e\n", largest_singval, smallest_singval, largest_singval/smallest_singval);

        while (smallest_singval<0.5*target_singval)
        {
            if (debug_print>=2)
                printf("@@@ +++ stabilizing smallest singval %e\n", smallest_singval);  
            singular_stabilize_U(smallest_singval, trailing_left_singvec, target_singval);
            smallest_singval = 1/power_iteration_for_singular_vect(UinvT, trailing_left_singvec, trailing_right_singvec, n_power_iterations);
        }

        while (largest_singval>2*target_singval)
        {
            if (debug_print>=2)
                printf("@@@ +++ stabilizing largest singval %e\n", largest_singval);  
            singular_stabilize_U(largest_singval, leading_left_singvec, target_singval);
            largest_singval = power_iteration_for_singular_vect(U, leading_left_singvec, leading_right_singvec, n_power_iterations);
        }


        // cudaDeviceSynchronize();
    }

    void singular_stabilize(real target_singval, int n_power_iterations=100) 
    {
        // singular_stabilize_OLDER(target_singval, n_power_iterations);
 
        if (n_power_iterations<=0) // use SVD rather than power iterations
            singular_stabilize_using_svd(target_singval); 
        else // use power_iteration
            singular_stabilize_using_power_iteration(target_singval, n_power_iterations); 
    }


    // Ensure numerical stability control through periodic renormalization or singuar stabilization or unfactorization
    void numerical_stability_control()
    {
        if (renormalize_period>0 && nupdates % renormalize_period==0)
        {
            if (debug_print>=1)
                printf("[R]");
            // output_layer.renormalize_U(udiag);
            renormalize_VT(1.0f);
            // printf("udiag = %e\n", udiag); printf("diag(U) = "); print(U.diag());
        }
        if (unfactorize_period>0 && nupdates % unfactorize_period==0)
        {
            if (debug_print>=1)
                printf("[U]");
            unfactorize(udiag);
        }
        if (stabilize_period>0 && nupdates % stabilize_period==0)
        {
            if (debug_print>=1)
                printf("[S]");

            if (debug_print>=2)
            {
                printf("@@@ BEGINNING singular_stabilize\n"); 
                printf("@@@ CHECKS BEFORE stabilization\n"); 
                print_scales();
                consistency_check();
            }

            singular_stabilize(udiag);

            if (debug_print>=2)
            { 
                printf("@@@ CHECKS AFTER stabilization\n"); 
                print_scales();
                consistency_check();
                printf("@@@ END singular_stabilize\n"); 
            }
        }
    }



// rank-m update to square matrix and to its inverse transpose, using the Woodbury identity
// Updates U and UinvT with the following U <- U + alpha A B^T  
// Based on Woodbury identity (internally performs inverse of a m x m matrix )
// B is a d x m matrix
void rankm_update_U_and_UinvT_Woodbury(real alpha, const CublasMat<real>& A, const CublasMat<real>& B, const CublasMat<real>& U, const CublasMat<real>& UinvT)
{
    int d = B.nrows();
    int m = B.ncols();

    // if(m>d)
    //    PLERROR("It makes no sense to update the inverse using Woodbury identity. It will be cheaper to compute the inverse of your updated U directly");


    // 7) U <- U + alpha A B^T
    blas_gemm_NT(alpha, A, B, 1, U);

    // 8) Corresponding update to UinvT

    // Compute Imm = ( B^T B + 1/alpha I )^-1
    // Note TODO: we could (and probably should) compute and invert a symmetric Imm (using only its lower or upper part)
    static CublasMat<real> Imm;
    Imm.resize(m,m);
    blas_gemm_TN(1, B, B, 0, Imm);
    add_scalar_to_diagonal(1/alpha, Imm);
    invertMatrix(Imm, (invup_mode<0));    

    // Compute B_Imm = B Imm   (a d x m matrix )
    static CublasMat<real> B_Imm;
    B_Imm.resize(d,m);
    blas_gemm_NN(1, B, Imm, 0, B_Imm);

    // Compute UinvT_B_Imm  (a d x m matrix )
    static CublasMat<real> UinvT_B_Imm;
    UinvT_B_Imm.resize(d,m);
    blas_gemm_NN(1, UinvT, B_Imm, 0, UinvT_B_Imm);
    
    // Perform update UinvT <- UinvT - UinvT_B_Imm B^T
    blas_gemm_NT(-1, UinvT_B_Imm, B, 1, UinvT);
}


// rankm_update_U_and_UinvT_recompute rank-m update to square matrix and recomputes its inverse from scratch.
// Performs:
// U <- U + alpha A B^T and correspondingly updates UinvT = U^-T
// U is a d x d matrix, UinvT is its inverse transposed
// A and B are d x k matrices.
// All matrices must be in column major mode

void rankm_update_U_and_UinvT_recompute(real alpha, const CublasMat<real>& A, const CublasMat<real>& B, const CublasMat<real>& U, const CublasMat<real>& UinvT)  // how big should this be????
{
    // 7) U <- U + alpha A B^T
    blas_gemm_NT(alpha, A, B, 1, U);

    // 8) Corresponding update to UinvT
    // We recompute the inverse of U from scratch
    //UinvT << U;
    // transpose_squaremat_inplace(UinvT);
    transpose(U, UinvT);
    invertMatrix(UinvT, (invup_mode<0));
}


// stream_for_H_tilde is the stream in which we scheduled computatiojn of A and in which we will update U 
void rankm_update_U_and_UinvT_Woodbury_streams(real alpha, const CublasMat<real>& A, const CublasMat<real>& B, const CublasMat<real>& U, const CublasMat<real>& UinvT, cudaStream_t stream_for_A_and_U)
{
    int d = B.nrows();
    int m = B.ncols();

    // if(m>d)
    //    PLERROR("It makes no sense to update the inverse using Woodbury identity. It will be cheaper to compute the inverse of your updated U directly");


    // 8) Corresponding update to UinvT

    // Compute Imm = ( B^T B + 1/alpha I )^-1
    // Note TODO: we could (and probably should) compute and invert a symmetric Imm (using only its lower or upper part)
    static CublasMat<real> Imm;
    Imm.resize(m,m);

    cublas.setStream(cublas.streams[0]); // do these in stream 0

    blas_gemm_TN(1, B, B, 0, Imm);
    add_scalar_to_diagonal(1/alpha, Imm);

    // now let's give the GPU some work in parallel so that the CPU can invert Imm in the meantime

    // Compute UinvT_B  (a d x m matrix )
    static CublasMat<real> UinvT_B;
    UinvT_B.resize(d,m);
    cublas.setStream(cublas.streams[1]); // let's do that in parallel
    blas_gemm_NN(1, UinvT, B, 0, UinvT_B);
    cublas.recordEvent(cuda_events[ISREADY_UinvT_B]);

    // 7) U <- U + alpha A B^T
    cublas.setStream(stream_for_A_and_U); // let's schedule  that in parallel
    blas_gemm_NT(alpha, A, B, 1, U);

    cublas.setStream(cublas.streams[0]); // switch back to stream 0
    cudaStreamSynchronize(cublas.streams[0]); // and wait for Imm to be ready
    invertMatrix(Imm, (invup_mode<0));                   // Host will compute the inverse

    // Make sure computation of UinvT_B is finished on the other stram
    cublas.waitEvent(cuda_events[ISREADY_UinvT_B]);
    // Now compute UinvT_B_Imm  (a d x m matrix )
    static CublasMat<real> UinvT_B_Imm;
    UinvT_B_Imm.resize(d,m);
    blas_gemm_NN(1, UinvT_B, Imm, 0, UinvT_B_Imm);
    
    // Perform update UinvT <- UinvT - UinvT_B_Imm B^T
    blas_gemm_NT(-1, UinvT_B_Imm, B, 1, UinvT);
}



// Conceptually Y, is a D x m matrix in column K-sparse representation, 
// so it is actually represented using two K x m matrices.
// Computation of grad_H can be turned off by passing an empty matrix as grad_H
// Computation of loss L can be turned off by passing a null pointer as L
// Computation+update of U and V can also be turned off by setting eta to 0.

void batch_fbpropupdate_using_Qtilde(const CublasMat<real>& H,        // d x m matrix of last hidden layer activations
                                            const CKSparseMat<real>& Y, 
                                            real eta, // learning_rate for parameter update
                                            // Outputs:
                                            real* L, // pointer to scalar total loss
                                            const CublasMat<real>& grad_H // d x m gradient on H 
    )

    {
        PLERROR("batch_fbpropupdate_using_Qtilde  not implemented. Using Q rathe than Qtilde is probably better anyway.");
    }




// Conceptually Y, is a D x m matrix in column K-sparse representation, 
// so it is actually represented using two K x m matrices.
// Computation of grad_H can be turned off by passing an empty matrix as grad_H
// Computation of loss L can be turned off by passing a null pointer as L
// Computation+update of U and V can also be turned off by setting eta to 0.

void batch_fbpropupdate_using_Q(const CublasMat<real>& H,        // d x m matrix of last hidden layer activations
                                const CKSparseMat<real>& Y, 
                                real eta, // learning_rate for parameter update
                                // Outputs:
                                real* L, // pointer to scalar total loss
                                const CublasMat<real>& grad_H // d x m gradient on H 
    )

// CublasMatrices are supposed to be in column-major order (fortran order).
 
    {
        static CublasMat<real> UT; // temporary (transpose of U) 

        bool compute_L = (L!=0);
        bool compute_grad_H = grad_H.isNotEmpty();

        // check dimensions of call arguments
        assert(H.nrows() == d && H.ncols() == m);
        assert(Y.indexes.nrows()==K && Y.indexes.ncols()==m);
        assert(Y.values.nrows()==K && Y.values.ncols()==m);
        assert( (grad_H.nrows() == d && grad_H.ncols() == m) || (!compute_grad_H) );

        // define work matrices views from TmpWork_d storage
        assert(TmpWork_d.nrows()==d && TmpWork_d.ncols()>=4*m);
        CublasMat<real> Y_hat = TmpWork_d.subMatColumns(0,m);    // d x m
        CublasMat<real> H_hat = TmpWork_d.subMatColumns(m,m);    // d x m
        CublasMat<real> Z_hat = TmpWork_d.subMatColumns(m+m,m);  // d x m
        CublasMat<real> H_tilde = TmpWork_d.subMatColumns(m+m+m,m); // d x m

        real two_eta = eta+eta;

        cudaDeviceSynchronize();   // Make sure everything launched in previous call has been updated 
        cublas.setStream(cublas.streams[0]); // switch to stream 0    
    
        // Compute H_hat = QH
        if(use_lower)
            blas_symm('L', 1, Q, H, 0, H_hat);
        else
            blas_gemm_NN(1, Q, H, 0, H_hat);

        // VT is a dense (d,D) matrix 
        // Y is a sparse (D,m) matrix in column-K-sparse format
        // H_tilde serves here as a temporary for storing V^T Y
        // product_dense_mat_cksparse_mat(VT, Y, H_tilde, cuda_streams, cuda_events);
        // product_dense_mat_cksparse_mat(VT, Y, H_tilde);
        product_dense_mat_cksparse_mat_using_gemmBatched(VT, Y, H_tilde);

        // In the mean time compute YTY on Host
        static BMat<real> YTY;
        YTY.resize(m,m);
        cksparse_square_mat(Y, YTY);

        // schedule async transfer of YTY into M in stream 3
        cublas.setStream(cublas.streams[3]); // becaus we will want to compute M and things that depend on it in stream 3
        YTY >> M;

        cublas.setStream(cublas.streams[0]); // switch back to stream 0
        // Finish computing Y_hat = U^T (V^T Y) = U^T H_tilde 
        // Y_hat will be a dense (d,m) matrix
        // cudaDeviceSynchronize(); // make sure H_tilde is ready
        blas_gemm_TN(1, U, H_tilde, 0, Y_hat);

        // Compute Z_hat = H_hat - Y_hat 
        blas_geam(1, H_hat, 'N', -1, Y_hat, 'N', Z_hat);
        // Z_hat << H_hat;
        // Z_hat -= Y_hat;
        cublas.recordEvent(cuda_events[ISREADY_Z_hat]);

        // Compute grad_H = 2 Z_hat
        if (compute_grad_H)
        {
            blas_geam(2, Z_hat, 'N', 0, grad_H, 'N', grad_H);
            cublas.recordEvent(cuda_events[ISREADY_grad_H]);
            // grad_H << Z_hat;
            // grad_H += Z_hat;
        }

        // Compute M = H^T Z_hat - Y_hat^T H + Y^T Y
        cublas.setStream(cublas.streams[3]); // let's compute M and things that depend on it in stream 3

        // first wait for Z_hat to be ready
        cublas.waitEvent(cuda_events[ISREADY_Z_hat]);
        blas_gemm_TN(1, H, Z_hat, 1, M);
        blas_gemm_TN(-1, Y_hat, H, 1, M);
        
        if (eta>0) // updating U, UinvT, Q and computing output H_hat that will be used to update V
        {

            // Now update Q (based on M) in stream 3 

            if(use_lower) // update only lower part of symmetric Q
            {
                // Q <- Q  - 2 eta ( H Z_hat^T + Z_hat H^T )
                blas_syr2k('L', -two_eta, H, Z_hat, 1, Q);

                // Q <- Q + (4 eta^2) (H M) H^T 
                blas_gemm_NN(two_eta*two_eta, H,M, 0, H_tilde);
                blas_gemm_NT(1, H_tilde, H, 1, Q); 
            }
            else // updating full Q
            {
                // Q <- Q - 2 eta H Z_hat^T
                blas_gemm_NT(-two_eta, H, Z_hat, 1, Q);

                // Q <- Q - 2 eta Z_hat H^T
                blas_gemm_NT(-two_eta, Z_hat, H, 1, Q);

                // Q <- Q + (4 eta^2) (H M) H^T 
                blas_gemm_NN(two_eta*two_eta, H,M, 0, H_tilde);
                blas_gemm_NT(1, H_tilde, H, 1, Q); 
            }


            // ******* Weight update a) of U and bookkeeping
            //  U <- U - 2 eta H_tilde H^T   (with H_tilde=UH ) and corresponding update to UinvT


            cublas.setStream(cublas.streams[0]);  // switch back to stream 0

            switch(std::abs(invup_mode))
            {
            case 1:    // in invup_mode 1, we don't keep an up-to date UinvT but will solve the linear system each time to compute UinvT H
            {
                blas_gemm_NN(1, U,H, 0, H_tilde);  // H_tilde <- UH

                // U <- U - 2 eta H_tilde H^T
                blas_gemm_NT(-two_eta, H_tilde, H, 1, U);

                // H_hat = 2 eta U^-T H
                // corresponds to solving for X:   U^T X = H  and setting H_hat = 2 eta X
                // Note: it woud probably be better to store and use U^T everywhere rather than U (like we store VT)
                UT.resize(d,d);
                transpose(U, UT);
                solveLinearSystem(UT, H_hat, H, (invup_mode<0));
                H_hat *= two_eta; // Compute H_hat = 2 eta UinvT H  

            }
                break;
            case 2:
                blas_gemm_NN(1, U,H, 0, H_tilde);  // H_tilde <- UH
                rankm_update_U_and_UinvT_iter_v1(-two_eta, H_tilde, H, U, UinvT);
                blas_gemm_NN(two_eta, UinvT, H, 0, H_hat); // Compute H_hat = 2 eta UinvT H  
                break;
            case 3:
                blas_gemm_NN(1, U,H, 0, H_tilde);  // H_tilde <- UH
                rankm_update_U_and_UinvT_iter_v2(-two_eta, H_tilde, H, U, UinvT);
                blas_gemm_NN(two_eta, UinvT, H, 0, H_hat); // Compute H_hat = 2 eta UinvT H  
                break;
            case 4:
                blas_gemm_NN(1, U,H, 0, H_tilde);  // H_tilde <- UH
                rankm_update_U_and_UinvT_Woodbury(-two_eta, H_tilde, H, U, UinvT);
                blas_gemm_NN(two_eta, UinvT, H, 0, H_hat); // Compute H_hat = 2 eta UinvT H  
                break;
            case 5:
                blas_gemm_NN(1, U,H, 0, H_tilde);  // H_tilde <- UH
                rankm_update_U_and_UinvT_recompute(-two_eta, H_tilde, H, U, UinvT);
                blas_gemm_NN(two_eta, UinvT, H, 0, H_hat); // Compute H_hat = 2 eta UinvT H  
                break;
            case 6:
            {
                cudaStream_t stream_for_H_tilde_and_update_of_U = cublas.streams[2]; // compute H_tilde in stream 2
                cublas.setStream(stream_for_H_tilde_and_update_of_U);  // schedule H_tilde <- UH in stream 2
                blas_gemm_NN(1, U,H, 0, H_tilde); 
                cublas.setStream(cublas.streams[0]);  // switch back to stream 0
                rankm_update_U_and_UinvT_Woodbury_streams(-two_eta, H_tilde, H, U, UinvT, stream_for_H_tilde_and_update_of_U);
                blas_gemm_NN(two_eta, UinvT, H, 0, H_hat); // Compute H_hat = 2 eta UinvT H  
            }
                break;
            default:
                PLERROR("Invalid abs(invup_mode) (must be between 1 and 6)");
            }
    
            // ******* Weight update b) of V and bookkeeping

            // Now using H_hat, we update matrix V (or more precisely V^T=VT)
            // VT <- VT + H_hat Y^T
            //   VT is a dense (d,D) matrix in column-major representation
            //   H_hat is a dense (d,m) matrix in column-major representation
            //   Y is a sparse (D,m) matrix in column-K-sparse representation
            //   so that Y^T is a (m,D) matrix 

            // printf("H_hat: %d x %d \n", H_hat.nrows(), H_hat.ncols() );

            // rank_update_dense_mat_cksparse_mat(H_hat, Y, VT, cuda_streams, cuda_events, cublas.streams[0], cublas.streams[0]); 
            rank_update_dense_mat_cksparse_mat_using_gemmBatched(H_hat, Y, VT);

        }

        // Compute loss L = Tr(M)
        if (compute_L) 
        {
            cublas.setStream(cublas.streams[3]); // we need M which is computed in stream 3
            *L = trace(M);
        }

        // outputs are L and grad_h
        // Make sure grad_H is done computing before the call returns
        if (compute_grad_H)
            cudaEventSynchronize(cuda_events[ISREADY_grad_H]);

    }

void batch_fbpropupdate_using_Q_OLD(const CublasMat<real>& H,        // d x m matrix of last hidden layer activations
                                const CKSparseMat<real>& Y, 
                                real eta, // learning_rate for parameter update
                                // Outputs:
                                real* L, // pointer to scalar total loss
                                const CublasMat<real>& grad_H // d x m gradient on H 
    )

// CublasMatrices are supposed to be in column-major order (fortran order).
 
    {
        static CublasMat<real> UT; // temporary (transpose of U) 

        bool compute_L = (L!=0);
        bool compute_grad_H = grad_H.isNotEmpty();

        // check dimensions of call arguments
        assert(H.nrows() == d && H.ncols() == m);
        assert(Y.indexes.nrows()==K && Y.indexes.ncols()==m);
        assert(Y.values.nrows()==K && Y.values.ncols()==m);
        assert( (grad_H.nrows() == d && grad_H.ncols() == m) || (!compute_grad_H) );

        // define work matrices views from TmpWork_d storage
        assert(TmpWork_d.nrows()==d && TmpWork_d.ncols()>=4*m);
        CublasMat<real> Y_hat = TmpWork_d.subMatColumns(0,m);    // d x m
        CublasMat<real> H_hat = TmpWork_d.subMatColumns(m,m);    // d x m
        CublasMat<real> Z_hat = TmpWork_d.subMatColumns(m+m,m);  // d x m
        CublasMat<real> H_tilde = TmpWork_d.subMatColumns(m+m+m,m); // d x m

        cudaDeviceSynchronize();
        cublas.setStream(cublas.streams[0]); // switch to stream 0    
        // cublas.setStream(0); // switch to stream 0    
    
        // Compute Y_hat = U^T (V^T Y) 
        // VT is a dense (d,D) matrix 
        // Y is a sparse (D,m) matrix in column-K-sparse format
        // Y_hat will be a dense (d,m) matrix
        // H_tilde serves here as a temporary for storing V^T Y

        // product_dense_mat_cksparse_mat(VT, Y, H_tilde); 
        product_dense_mat_cksparse_mat_using_gemmBatched(VT, Y, H_tilde); 
        // product_dense_mat_cksparse_mat(VT, Y, H_tilde, cuda_streams, cuda_events); 
        // cudaDeviceSynchronize();

        blas_gemm_TN(1, U, H_tilde, 0, Y_hat);

        // Compute H_hat = QH
        if(use_lower)
            blas_symm('L', 1, Q, H, 0, H_hat);
        else
            blas_gemm_NN(1, Q, H, 0, H_hat);

        // Compute Z_hat = H_hat - Y_hat 
        blas_geam(1, H_hat, 'N', -1, Y_hat, 'N', Z_hat);
        // Z_hat << H_hat;
        // Z_hat -= Y_hat;

        // Compute grad_H = 2 Z_hat
        if (compute_grad_H)
        {
            blas_geam(2, Z_hat, 'N', 0, grad_H, 'N', grad_H);
            // grad_H << Z_hat;
            // grad_H += Z_hat;
        }

        // Compute M = H^T Z_hat - Y_hat^T H + Y^T Y
        static BMat<real> YTY;
        YTY.resize(m,m);
        cksparse_square_mat(Y, YTY);
        YTY >> M;
        blas_gemm_TN(1, H, Z_hat, 1, M);
        blas_gemm_TN(-1, Y_hat, H, 1, M);
        
        // Compute loss L = Tr(M)
        if (compute_L) 
        {
            *L = trace(M);
        }

        if (eta>0) // updating U, UinvT, Q and computing output H_hat that will be used to update V
        {

        // H_tilde <- UH
        // we need H_tilde in all cases
        blas_gemm_NN(1, U,H, 0, H_tilde);

            // ******* Weight update a) of U and bookkeeping
            real two_eta = eta+eta;
            //  U <- U - 2 eta H_tilde H^T   (with H_tilde=UH ) and corresponding update to UinvT

            switch(std::abs(invup_mode))
            {
            case 0:    // in invup_mode 1, we don't keep an up-to date UinvT but will solve the linear system each time to compute UinvT H
            {
                // U <- U - 2 eta H_tilde H^T
                blas_gemm_NT(-two_eta, H_tilde, H, 1, U);

                // H_hat = 2 eta U^-T H
                // corresponds to solving for X:   U^T X = H  and setting H_hat = 2 eta X
                // Note: it woud probably be better to store and use U^T everywhere rather than U (like we store VT)
                UT.resize(d,d);
                transpose(U,UT);
                solveLinearSystem(UT, H_hat, H, (invup_mode<0));
                H_hat *= two_eta;
                // we define H_hat = 2 eta UinvT H  

            }
                break;
            case 1:
                rankm_update_U_and_UinvT_iter_v1(-two_eta, H_tilde, H, U, UinvT);
                break;
            case 2:
                rankm_update_U_and_UinvT_iter_v2(-two_eta, H_tilde, H, U, UinvT);
                break;
            case 3:
                rankm_update_U_and_UinvT_Woodbury(-two_eta, H_tilde, H, U, UinvT);
                break;
            case 4:
                rankm_update_U_and_UinvT_recompute(-two_eta, H_tilde, H, U, UinvT);
                break;
            default:
                PLERROR("Invalid abs(invup_mode) (must be between 1 and 4)");
            }
    
            // ******* Weight update b) of V and bookkeeping
            // we define H_hat = 2 eta UinvT H  

            if (invup_mode!=1)  // in invup_mode 0, we don't keep an up-to date UinvT but solve the linear system each time to compute UinvT H
                blas_gemm_NN(two_eta, UinvT, H, 0, H_hat);

            // Now using H_hat, we update matrix V (or more precisely V^T=VT)
            // VT <- VT + H_hat Y^T
            //   VT is a dense (d,D) matrix in column-major representation
            //   H_hat is a dense (d,m) matrix in column-major representation
            //   Y is a sparse (D,m) matrix in column-K-sparse representation
            //   so that Y^T is a (m,D) matrix 

            // printf("H_hat: %d x %d \n", H_hat.nrows(), H_hat.ncols() );

            // rank_update_dense_mat_cksparse_mat(H_hat, Y, VT, cuda_streams, cuda_events); 
            rank_update_dense_mat_cksparse_mat_using_gemmBatched(H_hat, Y, VT); 

            // printf("********* Before Q rank_update_dense_mat_cksparse_mat(H_hat, Y, VT))\n");

            // Now update Q
            if(use_lower) // update only lower part of symmetric Q
            {
                // Q <- Q  - 2 eta ( H Z_hat^T + Z_hat H^T )
                blas_syr2k('L', -two_eta, H, Z_hat, 1, Q);

                // Q <- Q + (4 eta^2) (H M) H^T 
                blas_gemm_NN(two_eta*two_eta, H,M, 0, H_tilde);
                blas_gemm_NT(1, H_tilde, H, 1, Q); 
            }
            else // updating full Q
            {
                // Q <- Q - 2 eta H Z_hat^T
                blas_gemm_NT(-two_eta, H, Z_hat, 1, Q);

                // Q <- Q - 2 eta Z_hat H^T
                blas_gemm_NT(-two_eta, Z_hat, H, 1, Q);

                // Q <- Q + (4 eta^2) (H M) H^T 
                blas_gemm_NN(two_eta*two_eta, H,M, 0, H_tilde);
                blas_gemm_NT(1, H_tilde, H, 1, Q); 
            }

        }
     
        // outputs are L and grad_h
        cudaStreamSynchronize(cublas.streams[0]); 
        // cublas.streams.synchronize();
        // cudaDeviceSynchronize();
    }

    

    virtual void batch_fbpropupdate(const CublasMat<real>& H,        // d x m matrix of last hidden layer activations
                                    const CKSparseMat<real>& Y,
                                    real eta, // learning_rate for parameter update
                                    // Outputs:
                                    real* L, // pointer to scalar total loss
                                    const CublasMat<real>& grad_H // d x m gradient on H 
        ) 
    {
        if (use_Qtilde) // call Q_tilde based algo
        {
            batch_fbpropupdate_using_Qtilde(H, Y, eta, L, grad_H );
        }
        else
        {
            batch_fbpropupdate_using_Q(H, Y, eta, L, grad_H );
        }

        if (eta>0)
        {
            nupdates++;
            numerical_stability_control();
        }
    }
    

// All matrices are supposed to be in column-major order
void online_fbpropupdate_using_Q(const CublasVec<real>& h,        // d-dimensional vector of last hidden layer
                                 const KSparseVec<real>& y,
                                 real eta, // learning_rate
                                 
                                 // computed outputs
                                 real* L, // pointer to scalar loss
                                 const CublasVec<real>& grad_h // gradient on h (d-dimensional vector)
    )
    {
        
        bool compute_L = (L!=0);
        bool compute_grad_h = grad_h.isNotEmpty();  

        // check dimensions of call arguments
        assert(h.length() == d);
        assert(y.indexes.length() == K);
        assert(y.values.length() == K);
        assert( (grad_h.length() == d)  || (!compute_grad_h) );
        // define work vector views from TmpWork_d storage 
        assert(m>0);
        assert(TmpWork_d.nrows()==d && TmpWork_d.ncols()>=4);
        CublasVec<real> y_hat = TmpWork_d.column(0);
        CublasVec<real> h_hat = TmpWork_d.column(1);
        CublasVec<real> z_hat = TmpWork_d.column(2);
        CublasVec<real> h_tilde = TmpWork_d.column(3);
  
        // h_hat = Qh
        if(use_lower)    
            blas_symv('L', 1, Q, h, 0, h_hat);
        else
            blas_gemv('N', 1, Q, h, 0, h_hat);

        // y_hat = U^T V^T y
        // (we use H_tilde for a temporary intermediate result)
        product_dense_mat_ksparse_vec(VT, y, h_tilde);
        blas_gemv('T', 1, U, h_tilde, 0, y_hat);
        
        // z_hat = h_hat - y_hat
        z_hat << h_hat;
        z_hat -= y_hat;

        if (compute_grad_h)
        {
            // grad_h = 2 z_hat
            grad_h << z_hat;
            grad_h *= 2;
        }

        // ******* Cost L
        // 5) L <- h^T h_hat - 2 h^T y_hat + sqnorm_y
        real L_first_term = blas_dot(h, h_hat);
        real L_second_term = blas_dot(h, y_hat);
        real sqnorm_y = ksparse_squared_norm(y.values, y.indexes);
        real loss =  L_first_term - (L_second_term+L_second_term) + sqnorm_y;
        // Debug printing
        // printf(">>> Decomposing computation of L (online blas version): L = %e - 2 * %e + %e = %e \n", L_first_term, L_second_term, sqnorm_y, *L);
        
        if (compute_L)  // Note: we need the loss anyway to update Q below
            *L = loss;


        if (eta>0) // updating U, UinvT, Q, and VT
        {
            // ******* Weight update a) of U and bookkeeping
            real two_eta = eta+eta;
  

            // U <- U - 2 eta (Uh) h^T            
            blas_gemv('N', 1, U, h, 0, h_tilde); // h_tilde = Uh
            blas_ger(-two_eta, h_tilde, h, U);

            // UinvT <- UinvT + (2 eta)/(1 - 2 eta ||h||^2 ) (UinvT h) h^T
            // We can also update det_U = (1 - 2 eta ||h||^2 ) det_U 
            real h_sqnorm = blas_dot(h,h);
            real coef = 1-two_eta*h_sqnorm;
            det_U *= coef;
            real alpha = two_eta/coef;
            blas_gemv('N', alpha, UinvT, h, 0, h_tilde); // we reuse h_tilde for another intermediate result 
            blas_ger(1, h_tilde, h, UinvT);
  

            // ******* Weight update b) of V and bookkeeping
            // we define h_tilde = 2 eta UinvT h 
            blas_gemv('N', two_eta, UinvT, h, 0, h_tilde);
            // V <- V + y h_tilde^T
            // or equiv VT <- VT + h_tilde y^T
            rank_one_update_dense_vec_ksparse_vec(h_tilde, y, VT);

            if(use_lower) // update only lower part of symmetric Q
            {
                // Q <- Q -2 eta (h z_hat^T + z_hat h^T)
                blas_syr2('L', -two_eta, h, z_hat, Q);
                // Q <- Q + (4 eta^2 L) h h^T
                blas_syr('L', two_eta*two_eta*loss, h, Q);
            }
            else // updating full Q
            {
                // Q <- Q -2 eta (h z_hat^T + z_hat h^T)
                blas_ger(-two_eta, h, z_hat, Q);
                blas_ger(-two_eta, z_hat, h, Q);
                // Q <- Q + (4 eta^2 L) h h^T
                blas_ger( two_eta*two_eta*loss, h, h, Q);
            }
        }

    }


// All matrices are supposed to be in column-major order
void online_fbpropupdate_using_Qtilde(const CublasVec<real>& h,        // d-dimensional vector of last hidden layer
                                             const KSparseVec<real>& y,
                                             real eta, // learning_rate
                                             
                                             // computed outputs
                                             real* L, // pointer to scalar loss
                                             const CublasVec<real>& grad_h // gradient on h (d-dimensional vector)
    )
    {
        PLERROR("online_fbpropupdate_using_Qtilde  not implemented. Using Q rathe than Qtilde is probably better anyway.");
    }

    virtual void online_fbpropupdate(const CublasVec<real>& h,        // d-dimensional vector of last hidden layer
                                     const KSparseVec<real>& y,
                                     real eta, // learning_rate
                                     
                                     // computed outputs
                                     real* L, // pointer to scalar loss
                                     const CublasVec<real>& grad_h // gradient on h (d-dimensional vector)
        ) 
    {
        if (use_Qtilde) // call Q_tilde based algo
        {
            online_fbpropupdate_using_Qtilde(h, y, eta, L, grad_h );
        }
        else
        {
            online_fbpropupdate_using_Q(h, y, eta, L, grad_h );
        }

        if (eta>0)
        {
            nupdates++;
            numerical_stability_control();
        }
    }

    virtual void print_scales() const
    {
        if (Q.isNotEmpty())
        {
            printf("Squared norms of W: (diagonal of Q)"); 
            print( Q.diagonal() );
        }
        if (Qtilde.isNotEmpty())
        {
            printf("Squared norms of V: (diagonal of Qtilde)"); 
            print( Qtilde.diagonal() );
        }
        printf("Updated determinant of U: %g\n", det_U); 
        printf("Diagonal of U: "); print( U.diagonal() );
        printf("Diagonal of UinvT: "); print( UinvT.diagonal() );
    }


//! Takes as input: non-factored parameters WT_ref as a reference (as it was before gradient update)
//! as well as WT, and factored parameters VT and U.
//! Computes corresponding WT_factored = U^T VT
//! And outputs statistics about the differences between these matrices.
//! U is a d x d matrix; All other matrices, including the temporary tmpmat must be d x D
//! if print_grad is true, the call will print things related to the gradient, specifically the difference between WT-WT_ref and WT_factored-WT_ref
void compare_parameters_with_nonfactored(const CublasMat<real>& WT_ref, const CublasMat<real>& WT, int print_grad=false)
{
  // compute WT_factored = U^T VT
    CublasMat<real> WT_factored(d,D);

  blas_gemm_TN(1, U, VT, 0, WT_factored);

  printf( "max_abs_diff(WT_ref, WT) = %e \n", max_abs_diff(WT_ref, WT) );
  printf( "max_abs_diff(WT_ref, WT_factored) = %e \n", max_abs_diff(WT_ref, WT_factored) );
  printf( "max_abs_diff(WT, WT_factored) = %e \n", max_abs_diff(WT, WT_factored) );

  if (print_grad)
  {
      CublasMat<real> tmpmat(d,D);

      printf("WT = "); print(WT);
      printf("WT_factored = "); print(WT_factored);
      
      diff(WT, WT_ref, tmpmat);
      printf("WT - WT_ref = "); print(tmpmat);

      diff(WT_factored, WT_ref, tmpmat);
      printf("WT_factored - WT_ref = "); print(tmpmat);
    }
}

}; // End of class definition


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
