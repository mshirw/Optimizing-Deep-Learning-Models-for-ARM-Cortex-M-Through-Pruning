#include <stdint.h>
#include <stdio.h>
#include "arm_math.h"
#include "cortexm_weight.h"
#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"
#include "cmsis_gcc.h"
#ifdef _RTE_
#include "RTE_Components.h"
#ifdef RTE_Compiler_EventRecorder
#include "EventRecorder.h"
#endif
#endif
#include "cortexm_main.h"
#include "matmul.h"
#include "add.h"
#if defined(ONNC_PC_SIM)
#else
	#include "mbed.h"
#endif

//#define ARM_MATH_DSP

Timer t;

#define CONV0_CHANNEL 8
#define CONV1_CHANNEL 16
#define FC0_WEIGHT_ALL 256
#define CONV0_BIAS_SHIFTRIGHT 0
#define CONV1_BIAS_SHIFTRIGHT 0
#define FC0_BIAS_SHIFTRIGHT 0

/*================================================*/

static q7_t conv0_wt[CONV0_CHANNEL*1*5*5] = CONV0_WEIGHT;
static q7_t conv0_bias[CONV0_CHANNEL] = CONV0_BIAS;

static q7_t conv1_wt[CONV1_CHANNEL*8*5*5] = CONV1_WEIGHT;
static q7_t conv1_bias[CONV1_CHANNEL] = CONV1_BIAS;

static q7_t fc0_wt[FC0_WEIGHT_ALL * 10] = FC0_WEIGHT;
static q7_t fc0_bias[10] = FC0_BIAS;



q7_t output_data[10];
q7_t col_buffer[2*5*5*32*2];
q15_t bufferA_conv0[2*3*3*3];
q15_t bufferA_conv1[2*CONV0_CHANNEL*3*3];

q7_t scratch_buffer[3*8*32*32];
q7_t scratch_buffer2[3*8*32*32];
q15_t fc0_buffer[FC0_WEIGHT_ALL];

bool bias_shiftright = false;
//Serial port2(USBTX, USBRX);

void fc_test(const q7_t* pV,
             const q7_t* pM,
             const uint16_t dim_vec,
             const uint16_t num_of_rows,
             const uint16_t bias_shift,
             const uint16_t out_shift, const q7_t* bias, q7_t* pOut, q15_t* vec_buffer)
{
    int       i, j;

    /* Run the following code as reference implementation for Cortex-M0 and Cortex-M3 */
    for (i = 0; i < num_of_rows; i++)
    {
        //int       ip_out = ((q31_t)(bias[i]) << bias_shift) + NN_ROUND(out_shift);
        int ip_out = 0;
        for (j = 0; j < dim_vec; j++)
        {
          ip_out += pV[j] * pM[i * dim_vec + j];
        }
        pOut[i] = (q7_t) __SSAT((ip_out >> out_shift), 8);
    }

}

void avepool_q7_HWC(q7_t * Im_in,
                   const uint16_t dim_im_in,
                   const uint16_t ch_im_in,
                   const uint16_t dim_kernel,
                   const uint16_t padding,
                   const uint16_t stride, const uint16_t dim_im_out, q7_t * bufferA, q7_t * Im_out)
{
    int16_t   i_ch_in, i_x, i_y;
    int16_t   k_x, k_y;

    for (i_ch_in = 0; i_ch_in < ch_im_in; i_ch_in++)
    {
        for (i_y = 0; i_y < dim_im_out; i_y++)
        {
            for (i_x = 0; i_x < dim_im_out; i_x++)
            {
                int       sum = 0;
                int       count = 0;
                for (k_y = i_y * stride - padding; k_y < i_y * stride - padding + dim_kernel; k_y++)
                {
                    for (k_x = i_x * stride - padding; k_x < i_x * stride - padding + dim_kernel; k_x++)
                    {
                        if (k_y >= 0 && k_x >= 0 && k_y < dim_im_in && k_x < dim_im_in)
                        {
                            sum += Im_in[i_ch_in + ch_im_in * (k_x + k_y * dim_im_in)];
                            count++;
                        }
                    }
                }
                Im_out[i_ch_in + ch_im_in * (i_x + i_y * dim_im_out)] = sum / count;
            }
        }
    }

}

q7_t* cortexm_main(int *image_data){

  #ifdef RTE_Compiler_EventRecorder
    EventRecorderInitialize (EventRecordAll, 1);
  #endif

  q7_t *img_buffer1 = scratch_buffer;
  q7_t *img_buffer2 = scratch_buffer2;

  for(int loop = 0 ; loop<784 ; loop++ ){
      img_buffer2[loop] = image_data[loop];
  }

  /*if (bias_shiftright == false){
    bias_shift();
    bias_shiftright = true;
  }*/
  
  t.start();

  arm_convolve_HWC_q7_basic(img_buffer2,28,1,conv0_wt,CONV0_CHANNEL,5,2,1,conv0_bias,0,CONV0_OUT_SHIFT,img_buffer1,28,(q15_t *)bufferA_conv0,NULL);

  arm_relu_q7(img_buffer1, 1 * CONV0_CHANNEL * 28 * 28 );

  arm_maxpool_q7_HWC(img_buffer1,28,CONV0_CHANNEL,2,0,2,14,NULL,img_buffer2);

  arm_convolve_HWC_q7_basic(img_buffer2,14,CONV0_CHANNEL,conv1_wt,CONV1_CHANNEL,5,2,1,conv1_bias,0,CONV1_OUT_SHIFT,img_buffer1,14,(q15_t *)bufferA_conv1,NULL);

  arm_relu_q7(img_buffer1,1 * CONV1_CHANNEL * 14 * 14);

  arm_maxpool_q7_HWC(img_buffer1,14,CONV1_CHANNEL,3,0,3,4,NULL,img_buffer2);

  fc_test(img_buffer2, fc0_wt, FC0_WEIGHT_ALL, 10, 0, FC0_OUT_SHIFT, fc0_bias, img_buffer1, fc0_buffer);
  t.stop();
  printf("This inference taken was %d ms.\n", t.read_ms());  
  t.reset();


  return img_buffer1;
}
