

#include <stdio.h>

#include "SparseTargetFactoredLinearOutputLayer.h"



PP<SparseTargetFactoredLinearOutputLayer> largesparse_cpu_get_layer(int d, int D, int K, int m, 
                                                                    float* V_data, int V_stride,
                                                                    float* U_data, int U_stride, 
                                                                    float* UinvT_data, int UinvT_stride,
                                                                    float* Q_data, int Q_stride,
                                                                    bool use_Qtilde = false,
                                                                    bool use_lower = true,
                                                                    int invup_mode = 0
                                                                    )
{
  static PP<SparseTargetFactoredLinearOutputLayer> layer = 0;

  if (layer==0)
    {
      layer = new SparseTargetFactoredLinearOutputLayer();
    }

  layer->d = d;
  layer->D = D;

  layer->U = BMat<real>(U_data, d, d, U_stride);
  layer->UinvT = BMat<real>(UinvT_data, d, d, UinvT_stride);
  layer->VT = BMat<real>(V_data, d, D, V_stride);

  // Use Q or Q_tilde ?
  layer->use_Qtilde = use_Qtilde;
  if (use_Qtilde)
    layer->Qtilde = BMat<real>(Q_data, d, d, Q_stride);  
  else
    layer->Q = BMat<real>(Q_data, d, d, Q_stride);  

  layer->use_lower = use_lower;
  layer->invup_mode = invup_mode;

  if (K>0)
    {
      layer->K = K;
      layer->m = m;
      layer->resize_work_storage();
    }
  
  return layer;
}


void largesparse_cpu_maintainance(int maintainance_op,
                                 int d, int D, 
                                 float* V_data, int V_stride,
                                 float* U_data, int U_stride, 
                                 float* UinvT_data, int UinvT_stride,
                                 float* Q_data, int Q_stride,
                                 bool use_Qtilde = false,
                                 bool use_lower = true,
                                 int invup_mode = 0)
  
{
  PP<SparseTargetFactoredLinearOutputLayer> layer 
    = largesparse_cpu_get_layer(
                                d, D, -1, -1, 
                                V_data, V_stride,
                                U_data, U_stride, 
                                UinvT_data, UinvT_stride,
                                Q_data, Q_stride,
                                use_Qtilde, use_lower, invup_mode);
  
  switch(maintainance_op)
    {
    case 0:
      layer->sanity_check_dimensions();
      break;
    case 1:
      layer->print_scales();
      break;
    case 2:
      layer->consistency_check();
      break;
    case 3:
      layer->renormalize_VT(1.0f);
      break;
    case 4:
      layer->singular_stabilize(1.0f);
      break;
    case 5:
      layer->unfactorize(1.0f);
      break;
    default:
      PLERROR("Invalid maintainance_op number");
    }

}

void largesparse_cpu_fbprop_update(int d, int D, int K, int m, 
                                  float* V_data, int V_stride,
                                  float* U_data, int U_stride, 
                                  float* UinvT_data, int UinvT_stride,
                                  float* Q_data, int Q_stride,
                                  float* H_data, int H_stride,
                                  int* Y_indexes_data, int Y_indexes_stride,
                                  float* Y_values_data, int Y_values_stride,
                                  float learning_rate,
                                  float* cost, 
                                  float* grad_H_data, int grad_H_stride,
                                  bool use_Qtilde = false,
                                  bool use_lower = true,
                                  int invup_mode = 0,

                                  int stabilize_period = 10,    // every how many updates to singular_stabilize
                                  int unfactorize_period = 100,  // every how many updates to unfactorize
                                  int debug_print = 0
                                  )
{
  PP<SparseTargetFactoredLinearOutputLayer> layer 
    = largesparse_cpu_get_layer(d, D, K, m, 
                                V_data, V_stride,
                                U_data, U_stride, 
                                UinvT_data, UinvT_stride,
                                Q_data, Q_stride,
                                use_Qtilde, use_lower, invup_mode);

//  if (layer->nupdates == 0 && grad_H_data != 0)
//    {
//      layer->sanity_check_dimensions();
//      layer->print_scales();
//      layer->consistency_check();
//    }

  layer->renormalize_period = 0;  // every how many updates to renormalize
  layer->unfactorize_period = unfactorize_period;  // every how many updates to unfactorize
  layer->stabilize_period = stabilize_period;  // every how many updates to singular_stabilize
  layer->debug_print = debug_print;

  // printf("n updates %d.", layer->nupdates);
  //if (layer->nupdates%100 == 0)
  //  {
  //    layer->singular_stabilize(1);
  //  }

  BMat<real> H(H_data, d, m, H_stride);

  BMat<real> grad_H(grad_H_data, grad_H_data!=0 ?d :0, grad_H_data!=0 ?m :0, grad_H_stride);
  BMat<int> Y_indexes(Y_indexes_data, K, m, Y_indexes_stride);
  BMat<real> Y_values(Y_values_data, K, m, Y_values_stride);  
  CKSparseMat<real> Y(D, Y_values, Y_indexes);

  layer->batch_fbpropupdate(H, Y, learning_rate, cost, grad_H);

}



/*
int main(int arg, char** argv)
{
  int d = 100;
  int D = 10000;
  int K = 2;
  int m = 20;
  
  float* V_data = 0;
  int V_stride = 0;
  float* U_data = 0;
  int U_stride = 0;
 
  float* UinvT_data = 0;
  int UinvT_stride = 0;
  float* Q_data = 0;
  int Q_stride = 0;
  float* H_data = 0;
  int H_stride = 0;
  int* Y_indexes_data = 0;
  int Y_indexes_stride = 0;
  float* Y_values_data = 0;
  int Y_values_stride = 0;
  float learning_rate = 0;
  float* cost = 0;
 
  float* grad_H_data = 0;
  int grad_H_stride = 0;
  

   largesparse_cpu_float(d,
                          D,
                          K,
                          m, 
                          V_data, V_stride,
                          U_data, U_stride, 
                          UinvT_data, UinvT_stride,
                          Q_data, Q_stride,
                          H_data, H_stride,
                          Y_indexes_data, Y_indexes_stride,
                          Y_values_data, Y_values_stride,
                          learning_rate,
                          cost, 
                          grad_H_data, grad_H_stride);
   

  return 0;
}
*/

