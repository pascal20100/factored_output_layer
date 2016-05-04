// -*- C++ -*-

// TFactoredSphericalLayer.h
// Copyright (C) 2016 Pascal Vincent and Universite de Montreal
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

#ifndef TFactoredSphericalLayer_INC
#define TFactoredSphericalLayer_INC

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>

#include <cstring>
#include <string>

#include <blas_linalg.h>
#include <k_sparse.h>
#include <PP.h>

using std::string;
using std::strcmp;
using namespace PLearn;


// #include <plearn/math/blas_proto.h>
// #include <plearn/math/lapack_proto.h>

// #define real float
#include "pl_math.h"

namespace PLearn {

template<typename RealMat, typename IntMat>
class TFactoredSphericalLayer:  public PPointable
{
public:
    typedef typename RealMat::elem_t real_t;
    typedef typename RealMat::vec_t RealVec;    
    typedef typename IntMat::vec_t IntVec;
    
    int d; // last hidden layer dimension
    int D; // output layer dimension
    int debug_print; // debug level

    bool use_lower; // if true, will use only lower triangular part of Q
    int invup_mode; // determines how we will update the inverse of U UinvT
    // invup_mode has the following meaning 
    //  1: don't kep an up-to-date UinvT, but solve a linear system each time
    //  2: iterate rank-1 updates; 
    //  3: iterate rank-one updates with special handling of first update (saves one matrix vector product)
    //  4: use Woodbury identity
    //  5: recompute full inverse

    // If this is set to true, then we will not be computing s, and have no omega, and no w_bar
    bool no_s_no_omega; 
    
    // Parameter initialization specification
    real_t udiag;
    real_t iVlow;
    real_t iVhigh; 
    int iVk;
    
    // Parameters for controling numerical stability 
    int nupdates; // Number of updates since initialization

    int unfactorize_period;  // every how many updates to unfactorize
    int stabilize_period;  // every how many updates to singular_stabilize

    // Model parameters U, V, omega (such that W = VU + 1_D omega^T )  
    // and bookkeeping matrices UinvT and Q and vector w_bar (all are updated by the call)
    RealMat U;         // d x d
    RealMat UinvT; // d x d This is up-to-date U^-T
    real_t det_U;         // scalar: determinant of U
    RealMat VT;    // d x D matrix VT = V^T
    RealMat Q;       // d x d this is up-to-date W^T W = U^T V^T V U
    
    RealVec omega;   // d
    RealVec w_bar;   // d dimensional vector: up to date sum of the D rows of W

    // Matrices and vectors for intermediate computations
    RealMat H_hat;    // d x m
    RealMat Z_hat;    // d x m
    RealMat H_tilde;  // d x m
    RealMat tempmat_dm; //  d x m
    RealMat two_H_diag_grad_q; // d x m

    RealVec h_tilde;  // m
    RealVec y_bar;  // m
    RealVec two_grad_q; // m

    RealMat M;          // m x m 
    RealMat YTY;        // m x m   YTY = Y_dot^T Y_dot 

public:
    
    //! Computes WT = U^T VT + omega
    void compute_WT(const RealMat& WT) const
    {
        blas_gemm_TN(1, U, VT, 0, WT);
        if(!no_s_no_omega)
            WT += omega;
    }

    // Allocates, computes and returns WT = U^T V^T
    virtual RealMat get_WT() const 
    {
        RealMat WT(d,D);
        compute_WT(WT);
        return WT;
    }

    void allocate_params()
    {
        U = RealMat(d,d);
        UinvT = RealMat(d,d);
        VT = RealMat(d,D);
        Q = RealMat(d,d);
        if(!no_s_no_omega)
        {
            omega = RealVec(d);
            w_bar = RealVec(d);
        }
    }
    
    void resize_work_storage(int m) 
    {
        assert(m>0);

        H_hat.resize(d, m);
        Z_hat.resize(d, m);
        H_tilde.resize(d, m);
        tempmat_dm.resize(d, m);
        two_H_diag_grad_q.resize(d, m);

        h_tilde.resize(m);
        y_bar.resize(m);
        two_grad_q.resize(m);

        M.resize(m, m);
        YTY.resize(m, m);
    }

    void sanity_check_params_dimensions() const
    {
        // Parameter and bookkeeping matrices
        assert( U.nrows() == d && U.ncols() == d );
        assert( UinvT.nrows() == d && UinvT.ncols() == d );
        assert( VT.nrows() == d && VT.ncols() == D );
        assert( Q.nrows()==d && Q.ncols()==d );
        if(!no_s_no_omega)
        {
            assert(omega.length()==d);
            assert(w_bar.length()==d);
        }
    }
    

//! Initializes parameters U and VT based on specified properties
//! udiag, iVlow, iVhigh, iVk
//! VT will be initialized as a d x D matrix with each column having iVk non-zero elements between iVlow and iVhigh
//! U will be initialized as a d x d diagonal matrix with value udiag on the diagonal;
    void initialize_params()
    {
        // initialize omega to 0
        if (!no_s_no_omega)
            omega.clear();

        // Initialize U and UinvT as d x d diagonal matrices.
        U.clear();
        add_scalar_to_diagonal(udiag, U);  // U has udiag as its diagonal values
        UinvT.clear();
        add_scalar_to_diagonal(1/udiag, UinvT); // UinvT has 1/udiag as its diagonal values
        // determinant of U
        det_U = (real_t) pow(udiag,d);
        
        // Initialize VT (d x D colum-major matrix) and associated Q = V^T V


        // number of non-zero elems to generate in each colum of VT is iVk not related with the K-sparsity of the target) 

        if (iVk<=0 || iVk>=d) // random initialize full VT 
        {
            fill_random_uniform_continuous(VT, iVlow, iVhigh);
            blas_syrk('L', 1, VT, 0, Q);  // only lower part of Q is computed.
            copy_lower_to_upper(Q);
        }

        else if (iVk==1)  // Let's initialise each column of VT as sparse with only one non-zero element 
        {
            // Note that this results in a diagonal Q
            VT.fill(0);
            Q.fill(0);
            double range = iVhigh-iVlow;
            for (int col=0; col<D; col++)
            {
                int i = rand()%d;
                real_t val = (real_t)(iVlow + range*(((double)rand())/RAND_MAX));
                VT(i, col) = val;
                Q(i,i) += val*val;
            }
        }
        else // Let's initialise each column of VT as sparse with number of non-zero elements beign iVk  
        {
            VT.fill(0);
            Q.fill(0);

            BVec<int> indexes(iVk);
            BVec<real_t> values(iVk);
      
            int col;
            real_t* VT_col = VT.data();
            for (col=0; col<D; col++, VT_col+=VT.stride() )
            {
                fill_random_uniform_discrete_noduplicate(indexes, 0, d-1);
                fill_random_uniform_continuous(values, iVlow, iVhigh);          
                int ii, jj;

                for(jj=0; jj<iVk; jj++) 
                {
                    int j = indexes[jj];
                    real_t val_j = values[jj];
                    VT_col[j] = val_j;

                    for (ii=0; ii<iVk; ii++)
                    {
                        int i = indexes[ii];
                        real_t val_i = values[ii];
                        Q(i,j) += val_i*val_j;
                    }
                }
            }

        }

        // At this stage we have initialized VT and correpsonding Q = V^T V
        // Note that Q = W^T W = U^T V^T V U = U^T Q U
        // With a diagonal U = udiag I this yields Q = (udiag*udiag) Q

        // Do we actually have a Q to compute?
        if (Q.isNotEmpty() ) // Yes
        {
            Q *= (udiag*udiag);
        }

        // initialize w_bar 
        // TODO: this is super inefficient, because written in a haste
        // we'd better keep track of this sum along the way
        if (!no_s_no_omega)
        {
            sum_rowwise(VT, w_bar);
            w_bar *= udiag;
        }
        
        /*
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
        sanity_check_params_dimensions();
    }

    // default constructor
    TFactoredSphericalLayer()
        :d(-1),
         D(-1),
         debug_print(0),
         use_lower(true), invup_mode(4), no_s_no_omega(false),
         udiag(1), iVlow(-1), iVhigh(1), iVk(1),
         nupdates(0), unfactorize_period(0), stabilize_period(0),
         det_U(-1)
    {}
    
    // convenience constructor that calls build()
    TFactoredSphericalLayer(int d_, int D_, 
                           real_t udiag_=1, real_t iVlow_=-1, real_t iVhigh_=1, int iVk_=1)
        :d(d_), 
         D(D_),
         debug_print(0),
         use_lower(true), invup_mode(4), no_s_no_omega(false),
         udiag(udiag_), iVlow(iVlow_), iVhigh(iVhigh_), iVk(iVk_),
         nupdates(0), unfactorize_period(0), stabilize_period(0),
         det_U(-1)
    {
        build();
    }

//! Applys a gradient update corresponding to a weight decay on W of strength lambda, using learning rate eta
//! This is done by rescaling U and UinvT
    virtual void apply_weight_decay_update(real_t lambda, real_t eta)
    {
        real_t s = 1-eta*lambda;
        U *= s;
        UinvT *= (1/s);
        if (!no_s_no_omega)
            omega *= s;
    }

    
//! This call multiplies the columns of V (the rows of VT) by the given scales (a d-dimensional vector)
//! It correspondingly downscales U so that W=VU is unchanged, and updates UinvT and Q to reflect these changes
//! Beware that scales is changed by the call (set to its inverse)
void scale_VT_rows(const RealVec& scales)
{
    // int d = U.nrows();

    // printf("Scaling rows of VT by "); print(scales); printf("/n");
    // VT = diag(diag(U)/udiag) VT 
    scale_rows(VT, scales);

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

//! Checks consistency of U and UinvT and of Q and VT
    virtual void consistency_check() const
    {
        RealMat C(d,d); // allocate temporary matrix in host memory
        // Compute C = U UinvT^T - I
        blas_gemm_NT(1, U, UinvT, 0, C);
        add_scalar_to_diagonal(-1, C); 
        printf( "Consistency checking U and UinvT: max abs (U UinvT^T - I) = %e  elements of U in range [%e, %e]\n", max_abs(C), min(U), max(U) );
        
        BMat<real_t> Ucopy(d,d); // allocate temporary matrix in host memory
        BMat<real_t> svd_U;
        BMat<real_t> svd_VT;
        BVec<real_t> svd_s;
        Ucopy << U;
        lapackSVD(Ucopy, svd_U, svd_s, svd_VT, 'A');
        if (debug_print>=2)
        {
            printf("Singular value spectrum of U: "); 
            print(svd_s);
        }
        printf("Product of singular values: %g\n", product(svd_s));
        printf("Condition number: %g / %g = %g \n", svd_s[0], svd_s[d-1], svd_s[0]/svd_s[d-1]);
        
        // Compute C = V^T V 
        blas_syrk('L', 1, VT, 0, C);
        // make sure it's symmetric
        copy_lower_to_upper(C);

        if (Q.isNotEmpty()) // Compare U^T C U = U^T V^T V U = W^T W to Q
        {
            if (use_lower) // only the lower part of Q has been updated
                copy_lower_to_upper(Q);  // symmetrize it for our comparisons

            RealMat CU(d,d); // allocate temporary matrix
            blas_gemm_NN(1, C, U, 0, CU);
            blas_gemm_TN(1, U, CU, 0, C);
            printf( "Consistency checking Q: max abs (U^T (V^T V) U - Q) = %e\n", max_abs_diff(C,Q) );
            RealMat WT = get_WT();
            blas_gemm_NT(1, WT, WT, 0, C);
            printf( "Consistency checking Q: max abs ( WT W - Q) = ( (U^T VT) (U^T VT)^T - Q ) = %e\n", max_abs_diff(C,Q) );
            printf( "Squared norm of first column of WT and first diaognal element of Q: %e, %e\n", blas_dot(WT.row(0),WT.row(0)), Q(0,0) );
        }
    }
    
//! unfactorizing: getting back to vanilla state where U = I
//! Since this call essentially computed W=VU+omega it is O(d^2 D)
//! This calls get_WT which allocates a temporary matrix of same dimensions as VT (i.e. d x D )
//! which is freed upon exiting
    void unfactorize()
    {
        VT << get_WT();

        if(!no_s_no_omega)
            omega.clear();

        // U = I  and UinvT = I
        U.clear();
        add_scalar_to_diagonal((real_t)1, U);  // U has udiag as its diagonal values
        UinvT << U;
        // determinant of U
        det_U = (real_t) pow(udiag, U.nrows());
    }

    //! Will bring back the given singular value of U associated with the gven singular vector
    //! back to being equal to target_singval
    void singular_stabilize_U(real_t singval, const RealVec& singvec, real_t target_singval)
    {
        // Bringing back the largest and smallest singular values to value target_singval
        real_t s = singval;
        const RealVec& u = singvec;

        real_t alpha = (target_singval-s)/s;

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

        real_t beta = -alpha / (1 + alpha);

        // Do U <-  U + alpha u (U^T u)^T
        RealVec UT_u(d);
        blas_gemv('T', 1, U, u, 0, UT_u);
        blas_ger(alpha, u, UT_u, U);

        // Do UinvT <- UinvT + beta u (U^-1 u)^T
        RealVec Uinv_u(d);
        blas_gemv('T', 1, UinvT, u, 0, Uinv_u);
        blas_ger(beta, u, Uinv_u, UinvT);

        // Do VT <- VT + beta u (V u)^T
        RealVec Vu(D);
        blas_gemv('T', 1, VT, u, 0, Vu);
        blas_ger(beta, u, Vu, VT);
        
        // In principle we don't need to update Q since W=VU is unchanged.
    }


    void print_svd_spectrum_of_U()
    {
        BMat<real_t> Ucopy(d,d); // allocate temporary matrix on CPU host
        
        // First compute svd
        BMat<real_t> svd_U;
        BMat<real_t> svd_VT;
        BVec<real_t> svd_s;
        Ucopy << U;  // copy matrix U (from cpu or GPU) to matrix Ucopy (on CPU)
        lapackSVD(Ucopy, svd_U, svd_s, svd_VT, 'A');
        printf("\n@@@+++ svd spectrum: ");
        print(svd_s);
    }


    // This call does a SVD of U followed by an update to V that is O(2 Dd)
    // It corrects the singular value of U that re too far out of range so that they are put back to being target_singval
    void singular_stabilize_using_svd(real_t target_singval) 
    {
        int d = U.nrows();
        BMat<real_t> Ucopy(d,d); // allocate temporary matrix
        
        // First compute svd
        BMat<real_t> svd_U;
        BMat<real_t> svd_VT;
        BVec<real_t> svd_s;
        Ucopy << U;
        lapackSVD(Ucopy, svd_U, svd_s, svd_VT, 'A');

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
            unfactorize();
        }
        else  // repeatedly call singular_stabilize_U
        {
            RealMat SVDU(d,d); // allocate temporary matrix (either on CPU or GPU)
            SVDU << svd_U;     // if svd_U was on CPU and RealMat are GPU, then we're copying it to GPU 
            for(int k=d-1; k>=0; k--)
            {
                if (svd_s[k]<0.5*target_singval || svd_s[k]>2*target_singval)
                {
                    singular_stabilize_U(svd_s[k], SVDU.column(k), target_singval);
                    if (debug_print>=2)
                        print_svd_spectrum_of_U();
                }
            }
        }

    }

    // same as above, but leading and trailing singular values and vectors are computed using power iteration instead of SVD
    void singular_stabilize_using_power_iteration(real_t target_singval, int n_power_iterations) 
    {
        // Find leading and trailing left singular vector of U (assosciated to largest singular value) by power iteration
        RealVec leading_left_singvec(U.nrows());
        RealVec leading_right_singvec(U.ncols());
        RealVec trailing_left_singvec(UinvT.nrows());
        RealVec trailing_right_singvec(UinvT.ncols());

        leading_left_singvec << U.column(0);
        leading_right_singvec << U.row(0);
        trailing_left_singvec << UinvT.column(0);
        trailing_right_singvec << UinvT.row(0);
        
        // Find trailing left singular vector of U (associated to smallest singular value) 
        // It is the same as the leading left singular vector of UinvT, which we will compute by power iteration 
        if(invup_mode!=5) // Let us periodically recompute UinvT because iterated sherman morrison or woodbury lead to divergence
        {
            UinvT << U;
            transpose_squaremat_inplace(UinvT);
            invertMatrix(UinvT);
        }

        real_t largest_singval = power_iteration_for_singular_vect(U, leading_left_singvec, leading_right_singvec, n_power_iterations);
        real_t smallest_singval = 1/power_iteration_for_singular_vect(UinvT, trailing_left_singvec, trailing_right_singvec, n_power_iterations);
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
        
    }

    void singular_stabilize(real_t target_singval, int n_power_iterations=100) 
    {
        if (n_power_iterations<=0) // use SVD rather than power iterations
            singular_stabilize_using_svd(target_singval); 
        else // use power_iteration
            singular_stabilize_using_power_iteration(target_singval, n_power_iterations); 
    }

    // Ensure numerical stability control through periodic singuar stabilization or unfactorization
    void numerical_stability_control()
    {
        if (unfactorize_period>0 && nupdates % unfactorize_period==0)
        {
            if (debug_print>=1)
                printf("[U]");
            unfactorize();
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

    void batch_fprop(
        const RealMat& H,        // d x m matrix of last hidden layer activations of each of the m examples in a minibatch
        const IntMat& Kindexes,  // K x m matrix containing, for each of the m examples, the indexes for which we want to compute the actual output activation 
        // Outputs:
        const RealMat& A,  // K x m matrix containing the specific linear activations that we wanted computed, as specified in Kindexes                  
        const RealVec& q, // vector of length m containing the sum of squared linear output (for each of the m examples in the minibatch)
        const RealVec& s // vector of length m containing the sum of linear outputs (for each of the m examples in the minibatch)
        )
    {
        // printf("\n@@@@@@@ FPROP \n");

        int m = H.ncols();
        int K = Kindexes.nrows();

        resize_work_storage(m); 

        // Compute H_hat = QH
        if(use_lower)
            blas_symm('L', 1, Q, H, 0, H_hat);
        else
            blas_gemm_NN(1, Q, H, 0, H_hat);
        
        // q = diaognal(H^T H_hat)
        blas_gemm_TN(1, H, H_hat, 0, M); // (for now M correponds to M_hat in the paper pseudocode)
        q << M.diagonal();
                                             
        // s = H^T w_bar
        if (!no_s_no_omega)
            blas_gemv('T', 1, H, w_bar, 0, s);

        // H_tilde = UH  (a d x m matrix)
        blas_gemm_NN(1, U,H, 0, H_tilde);

        // Now compute A (a K x m matrix)
        // A_kj = H_tilde_j^T VT_Kindexes(k,j)
        product_selectedColumnsT_dense(VT, Kindexes, H_tilde, A);

        /*
        for(int j=0; j<m; j++)
        {
            RealVec H_tilde_j = H_tilde.column(j);
            for (int k=0; k<K; k++)
            {
                // A_kj = H_tilde_j^T VT_Kindexes(k,j)
                A(k,j) = blas_dot(H_tilde_j, VT.column( Kindexes(k,j) ) );
            }
        }
        */

        if (!no_s_no_omega)
        {
            // h_tilde = H^T omega
            blas_gemv('T', 1, H, omega, 0, h_tilde);
            
            addRowVector(A, h_tilde);
        }
    }

    void batch_bprop_update(
        real_t eta,         // the learning rate with which to update the model's parameters
        const RealMat& H,        // d x m matrix of last hidden layer activations of each of the m examples in a minibatch
        const IntMat& Kindexes,  // K x m matrix containing, for each of the m examples, the indexes for which we want to compute the actual output activation 
        const RealMat& grad_A,  // K x m matrix containing the gradients with respect to A i.e. to the specific linear activations that we computed, as specified by Kindexes                  
        const RealVec& grad_q, // vector of length m containing the gradient with respect to q (the sum of squared linear outputs for each of the m examples in the minibatch)
        const RealVec& grad_s, // vector of length m containing the gradient with respect to s (the sum of linear outputs for each of the m examples in the minibatch)
        // Output:
        const RealMat& grad_H, // d x m matrix containig gradients with respect to H
        // optional parameter
        bool accumulate_gradients_in_grad_H = false // specifies whether we should accumulate into (+=) rather than overwrite grad_H                      
    )
    {

        // printf("\n@@@@@@@ BPROP \n");
        int m = H.ncols();
        int K = Kindexes.nrows();
        resize_work_storage(m); 

        /*
        printf("\ngrad_A: ");
        print(grad_A);
        printf("\ngrad_q: ");
        print(grad_q);
        printf("\ngrad_s: ");
        print(grad_s);
        printf("\n");
        */

        // ----------------------------
        // y_bar = rowsum(grad_A);
        //fprintf(stderr, "grad_A: %d x %d\n",grad_A.nrows(),grad_A.ncols());
        //fprintf(stderr, "y_bar: %d\n",y_bar.length());
        if (!no_s_no_omega)
            sum_columnwise(grad_A, y_bar);

        // ----------------------------
        // Y_dot is a D x m column-K-sparse mat containing values of grad_A at positions given by Kindexes
        BMat<real_t> grad_A_copy(grad_A.nrows(), grad_A.ncols()); // copy grad_A to CPU matrix
        grad_A_copy << grad_A;
        CKSparseMat<real_t> Y_dot(D, grad_A_copy, Kindexes);

        // ----------------------------
        // Compute Z_hat = U^T (V^T Y_dot) + omega y_bar^T + w_bar grad_s^T
        // VT is a dense (d,D) matrix 
        // Y_dot is a sparse (D,m) matrix in column-K-sparse format
        // Z_hat will be a dense (d,m) matrix
        // we use a temporary d x m matrix for storing V^T Y
        product_dense_mat_cksparse_mat(VT, Y_dot, tempmat_dm); 
        blas_gemm_TN(1, U, tempmat_dm, 0, Z_hat);
        if (!no_s_no_omega)
        {
            blas_ger(1, omega, y_bar, Z_hat);
            blas_ger(1, w_bar, grad_s, Z_hat);
        }
        // ----------------------------
        // grad_H = 2 H_hat diag(grad_q) + Z_hat
        two_grad_q << grad_q;
        two_grad_q += two_grad_q;
        if (accumulate_gradients_in_grad_H)
            grad_H += H_hat;
        else
            grad_H << H_hat;
        scale_columns(grad_H, two_grad_q);
        grad_H += Z_hat;

        // two_H_diag_grad_q = 2 H diag(grad_q)
        scale_columns(H, two_grad_q, two_H_diag_grad_q);

        // ----------------------------
        // Computing M 
        // First make sure that M <- 4 diag(grad_q) H^T H_hat diag(grad_q)
        // we initially have M <- H^T H_hat (computed in beginning of fprop)
        // so it suffices to rescale its rows and columns
        scale_columns(M, two_grad_q);
        scale_rows(M, two_grad_q);
        // Now do M += D grad_s grad_s^T + Y_dot^T Y_dot + grad_s y_bar^T + y_bar grad_s^T
        if(use_lower)
        {
            // M += D grad_s grad_s^T
            if (!no_s_no_omega)
                blas_syr('L', D, grad_s, M);
            // M += Y_dot^T Y_dot
            cksparse_square_mat(Y_dot, YTY);
            M += YTY;
            // M += grad_s y_bar^T + y_bar grad_s^T
            if (!no_s_no_omega)
                blas_syr2('L', 1, grad_s, y_bar, M);
            // M += 2 diag(grad_q) H^T Z_hat  +  2 Z_hat^T H diag(grad_q)
            blas_syr2k_T('L', 1, two_H_diag_grad_q, Z_hat, 1, M);
        }
        else
        {
            PLERROR("Only implemented for case use_lower=true for now");
        }

        // ----------------------------
        // Update U and UinvT
        // Update U <- U -2 eta (UH) diag(grad_q) H^T
        // or equivalently U <- U -eta UH (2 H diag(grad_q) )^T
        // since we alreasdy have UH in H_tilde and 2 H diag(grad_q) in  two_H_diag_grad_q
        // this amounts to update:
        // U <- U + (-eta) H_tilde two_H_diag_grad_q^T
        switch(invup_mode)
        {
        case 1:    // in invup_mode 0, we don't keep an up-to date UinvT (as we will solve the linear system each time to compute UinvT H)
            blas_gemm_NT(-eta, H_tilde, two_H_diag_grad_q, 1, U);
            break;
        case 2:
            rankm_update_U_and_UinvT_iter_v1(-eta, H_tilde, two_H_diag_grad_q, U, UinvT);
            break;
        case 3:
            rankm_update_U_and_UinvT_iter_v2(-eta, H_tilde, two_H_diag_grad_q, U, UinvT);
            break;
        case 4:
            rankm_update_U_and_UinvT_Woodbury(-eta, H_tilde, two_H_diag_grad_q, U, UinvT, H);
            break;
        case 5:
            rankm_update_U_and_UinvT_recompute(-eta, H_tilde, two_H_diag_grad_q, U, UinvT);
            break;
        default:
            PLERROR("Invalid invup_mode " << invup_mode << " (must be between 1 and 5)");
        }
        
        // ----------------------------
        if (!no_s_no_omega) // update omega
        {
            // Update omega <- omega - eta H ( 2 diag(grad_q) H^T omega + grad_s )
            // We already have initially h_tilde = H^T omega
            h_tilde *= two_grad_q;
            h_tilde += grad_s; // now h_tilde =  2 diag(grad_q) H^T omega + grad_s
            blas_gemv('N', -eta, H, h_tilde , 1, omega);
        }
        
        // ----------------------------
        // Updating VT <- VT - eta (UinvT H) Y_dot^T 
        // we will recycle H_hat, to H_hat = -eta UinvT H  
        if (invup_mode==1)  // in invup_mode 1, we don't maintain UinvT but solve the linear system each time to compute UinvT H
        {
            // alias UinvT as UT
            const RealMat& UT = UinvT;
            UT << U;
            transpose_squaremat_inplace(UT);
            solveLinearSystem(UT, H_hat, H);
            H_hat *= (-eta);
        }
        else //  compute H_hat = -eta UinvT H 
        {
            blas_gemm_NN(-eta, UinvT, H, 0, H_hat);
        }
        // Now using H_hat, we update matrix V (or more precisely V^T=VT)
        // VT <- VT + H_hat Y_dot^T
        //   VT is a dense (d,D) matrix in column-major representation
        //   H_hat is a dense (d,m) matrix in column-major representation
        //   Y is a sparse (D,m) matrix in column-K-sparse representation
        //   so that Y^T is a (m,D) matrix 

        // printf("********* Before Q rank_update_dense_mat_cksparse_mat(H_hat, Y, VT))\n");
        // printf("H_hat: %d x %d \n", H_hat.nrows(), H_hat.ncols() );
        rank_update_dense_mat_cksparse_mat(H_hat, Y_dot, VT); 


        // ----------------------------
        // Now update Q <- Q - eta grad_H H^T -eta H grad_H^T + eta^2 (HM)H^T
        if(use_lower) // update only lower part of symmetric Q
        {
            // Q <- Q  - eta  grad_H H^T -eta H grad_H^T
            blas_syr2k('L', -eta, grad_H, H, 1, Q);
            
            // Q <- Q + eta^2 (H M) H^T 
            blas_symm_rightside('L', eta*eta, M,H, 0, H_tilde);
            blas_gemm_NT(1, H_tilde, H, 1, Q); 
        }
        else
        {
            PLERROR("Only implemented for case use_lower=true for now");
        }


        // ----------------------------
        // Finally update w_bar <- w_bar - eta H (2 diag(grad_q) H^T w_bar + D grad_s + y_bar )
        // We'll use y_bar as our working vector

        if (!no_s_no_omega)  // no need to keep an up-to-date w_bar if we have no s and no omega
        {
            // First do y_bar += D grad_s
            blas_axpy( D, grad_s, y_bar );
            // Now do: y_bar += two_H_diag_grad_q ^T w_bar
            blas_gemv('T', 1, two_H_diag_grad_q, w_bar, 1, y_bar);
            // Finally do: w_bar <- w_bar - eta H y_bar
            blas_gemv('N', -eta, H, y_bar, 1, w_bar);
        }

        // Perform numerical stability control
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
            if (debug_print>=1)
            {
                printf("Squared norms of W: (diagonal of Q)"); 
                print( Q.diagonal() );
            }
            if (debug_print>=2)
            {
                // costly
                RealMat WT = get_WT();
                printf("Recomputed squared norm of W.column(0): %e \n", blas_dot(WT.row(0), WT.row(0)) );
            }
        }
        printf("Updated determinant of U: %g\n", det_U); 
        if (debug_print>=2)
        {
            printf("Diagonal of U: "); print( U.diagonal() );
            printf("Diagonal of UinvT: "); print( UinvT.diagonal() );
        }
    }
    
// -----------------------------------------
// low-level function API 


void setup_parameters_from_raw_pointers(
    int d, int D, 
    real_t* V_data, int V_stride,
    real_t* U_data, int U_stride, 
    real_t* UinvT_data, int UinvT_stride,
    real_t* Q_data, int Q_stride,
    real_t* omega_data, int omega_step,
    real_t* w_bar_data, int w_bar_step)
{
    this->d = d;
    this->D = D;

    U = RealMat(U_data, d, d, U_stride);
    UinvT = RealMat(UinvT_data, d, d, UinvT_stride);
    VT = RealMat(V_data, d, D, V_stride);    
    Q = RealMat(Q_data, d, d, Q_stride);
    if (omega_data==0 || w_bar_data==0)
    {
        no_s_no_omega = true;
    }
    else
    {
        no_s_no_omega = false;
        omega = RealVec(omega_data, d, omega_step);
        w_bar = RealVec(w_bar_data, d, w_bar_step);
    }
}

    
void setup_work_matrices_from_raw_pointers(
    int d, int m,
    real_t* work_d_5m, int work_d_5m_stride,
    real_t* work_m_3p2m, int work_m_3p2m_stride)
{
    this->d = d;
    this->D = D;

    // setup work state for inermediate computations
    // using work_d_5m to set up d x m matrices
    H_hat = RealMat(work_d_5m, d, m, work_d_5m_stride);
    work_d_5m += m * work_d_5m_stride;
    Z_hat = RealMat(work_d_5m, d, m, work_d_5m_stride);
    work_d_5m += m * work_d_5m_stride;
    H_tilde = RealMat(work_d_5m, d, m, work_d_5m_stride);
    work_d_5m += m * work_d_5m_stride;
    tempmat_dm = RealMat(work_d_5m, d, m, work_d_5m_stride);
    work_d_5m += m * work_d_5m_stride;
    two_H_diag_grad_q = RealMat(work_d_5m, d, m, work_d_5m_stride);
    // using work_m_3p2m to set up vectors of length m and m x m matrices
    h_tilde = RealVec(work_m_3p2m, m, 1);
    work_m_3p2m += work_m_3p2m_stride;
    y_bar = RealVec(work_m_3p2m, m, 1);
    work_m_3p2m += work_m_3p2m_stride;
    two_grad_q = RealVec(work_m_3p2m, m, 1);
    work_m_3p2m += work_m_3p2m_stride;
    M = RealMat(work_m_3p2m, m, m, work_m_3p2m_stride);
    work_m_3p2m += m * work_m_3p2m_stride;
    YTY = RealMat(work_m_3p2m, m, m, work_m_3p2m_stride);
    work_m_3p2m += m * work_m_3p2m_stride;    
}

    
static void stateless_fprop(
    // dimensions
    int d, int D, int m, int K,

    // model parameters and bookkeeping variables
    real_t* V_data, int V_stride,
    real_t* U_data, int U_stride, 
    real_t* UinvT_data, int UinvT_stride,
    real_t* Q_data, int Q_stride,
    real_t* omega_data, int omega_step,
    real_t* w_bar_data, int w_bar_step,

    // workspace 
    real_t* work_d_5m, int work_d_5m_stride,
    real_t* work_m_3p2m, int work_m_3p2m_stride,

    // minibatch value inputs
    real_t* H_data, int H_stride,
    int* Kindexes_data, int Kindexes_stride,
    
    // minibatch value outputs:
    real_t* A_data, int A_stride,
    real_t* q_data, int q_step,
    real_t* s_data, int s_step,

    // extra control parameters
    int use_lower = 1,
    int debug_print = 0
    )
{
    //fprintf(stderr, "fprop_Factored d=%d, D=%d \n", d, D);

    TFactoredSphericalLayer layer;
    
    layer.setup_parameters_from_raw_pointers(
        d, D, 
        V_data, V_stride,
        U_data, U_stride, 
        UinvT_data, UinvT_stride,
        Q_data, Q_stride,
        omega_data, omega_step,
        w_bar_data, w_bar_step);
    
    layer.setup_work_matrices_from_raw_pointers(
        d, m,
        work_d_5m, work_d_5m_stride,
        work_m_3p2m, work_m_3p2m_stride);
    
    // set other simple parameters
    layer.use_lower = (use_lower!=0);
    layer.debug_print = debug_print;    
    
    // minibatch value inputs
    RealMat H(H_data, d, m, H_stride);
    IntMat Kindexes(Kindexes_data, K, m, Kindexes_stride);
    
    // minibatch value outputs
    RealMat A(A_data, K, m, A_stride);
    RealVec q(q_data, m, q_step);
    RealVec s(s_data, m, s_step);

    layer.batch_fprop(H, Kindexes, A, q, s);
}

static void stateless_bprop_update(
    // dimensions
    int d, int D, int m, int K,

    // model parameters and bookkeeping variables
    real_t* V_data, int V_stride,
    real_t* U_data, int U_stride, 
    real_t* UinvT_data, int UinvT_stride,
    real_t* Q_data, int Q_stride,
    real_t* omega_data, int omega_step,
    real_t* w_bar_data, int w_bar_step,

    // workspace 
    real_t* work_d_5m, int work_d_5m_stride,
    real_t* work_m_3p2m, int work_m_3p2m_stride,

    // minibatch value inputs
    real_t* H_data, int H_stride,
    int* Kindexes_data, int Kindexes_stride,

    // minibatch gradient inputs
    real_t* grad_A_data, int grad_A_stride,
    real_t* grad_q_data, int grad_q_step,
    real_t* grad_s_data, int grad_s_step,

    // minibatch gradient output
    real_t* grad_H_data, int grad_H_stride,

    // bprop and update control parameters
    real_t eta,
    int accumulate_gradients_in_grad_H = 0,

    int use_lower = 1,
    int invup_mode = 0,
    int debug_print = 0,
    int run_stability_control = 0
    )
{
    TFactoredSphericalLayer layer;
    
    layer.setup_parameters_from_raw_pointers(
        d, D, 
        V_data, V_stride,
        U_data, U_stride, 
        UinvT_data, UinvT_stride,
        Q_data, Q_stride,
        omega_data, omega_step,
        w_bar_data, w_bar_step);
    
    layer.setup_work_matrices_from_raw_pointers(
        d, m,
        work_d_5m, work_d_5m_stride,
        work_m_3p2m, work_m_3p2m_stride);
    
    // set other simple parameters
    layer.use_lower = (use_lower!=0);
    layer.debug_print = debug_print;        
    
    // minibatch value inputs
    RealMat H(H_data, d, m, H_stride);
    IntMat Kindexes(Kindexes_data, K, m, Kindexes_stride);
    
    // minibatch gradient inputs
    RealMat grad_A(grad_A_data, K, m, grad_A_stride);
    RealVec grad_q(grad_q_data, m, grad_q_step);
    RealVec grad_s(grad_s_data, m, grad_s_step);

    // minibatch gradient output
    RealMat grad_H(grad_H_data, d, m, grad_H_stride);

    layer.invup_mode = invup_mode;
    layer.unfactorize_period = 0;  // every how many updates to unfactorize
    layer.stabilize_period = 0;  // every how many updates to singular_stabilize

    layer.batch_bprop_update(eta, H, Kindexes, grad_A, grad_q, grad_s,
                             grad_H, (accumulate_gradients_in_grad_H!=0) );

    if (run_stability_control>0)
    {
        layer.singular_stabilize(1,100);
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
