//
//  lognhmmgmm.h
//  simpleCNN
//
//  Created by mzhang on 14/12/10.
//  Copyright (c) 2014年 ___STAR___. All rights reserved.
//

#ifndef __simpleCNN__lognhmmgmm__
#define __simpleCNN__lognhmmgmm__

#include <stdio.h>
#include "stdlib.h"
#include "math.h"
#include "string.h"


#define STATE_NUM 15
#define GMM_SIZE 8
#define DIM_NUM 14

#define MINIMUM_V 1e-32
#define LOGMINIMUM -16384 //-1024 * 16

typedef struct impLogGSM_S
{
    const float *pfMean;
    const float fFactor;
    const float *pfInvCovar;
} IMP_LOGGSM_S;


typedef struct impLogGMM_S
{
    const float *pfWeights;
    IMP_LOGGSM_S astLogGSM[GMM_SIZE];
} IMP_LOGGMM_S;


typedef struct impLogNHMMGMM_Store_S
{
    const float *pfPrior;
    const float *pfTransfer;
    IMP_LOGGMM_S astLogGMMList[STATE_NUM];
} IMP_LOGNHMMGMM_STORE_S;


typedef struct impLogNHMMGMM_S
{
    int dwStateNum;
    float *pfPrior;
    float *pfTransfer;
} IMP_LOGNHMMGMM_S;


extern const float gafPriorState[15];
extern const float gafTransmat[225];
extern const float gafWgtList_0[8];
extern const float gafMean_0_0[14];
extern const float gfFactor_0_0;
extern const float gafInvCovar_0_0[196];
extern const float gafMean_0_1[14];
extern const float gfFactor_0_1;
extern const float gafInvCovar_0_1[196];
extern const float gafMean_0_2[14];
extern const float gfFactor_0_2;
extern const float gafInvCovar_0_2[196];
extern const float gafMean_0_3[14];
extern const float gfFactor_0_3;
extern const float gafInvCovar_0_3[196];
extern const float gafMean_0_4[14];
extern const float gfFactor_0_4;
extern const float gafInvCovar_0_4[196];
extern const float gafMean_0_5[14];
extern const float gfFactor_0_5;
extern const float gafInvCovar_0_5[196];
extern const float gafMean_0_6[14];
extern const float gfFactor_0_6;
extern const float gafInvCovar_0_6[196];
extern const float gafMean_0_7[14];
extern const float gfFactor_0_7;
extern const float gafInvCovar_0_7[196];
extern const float gafWgtList_1[8];
extern const float gafMean_1_0[14];
extern const float gfFactor_1_0;
extern const float gafInvCovar_1_0[196];
extern const float gafMean_1_1[14];
extern const float gfFactor_1_1;
extern const float gafInvCovar_1_1[196];
extern const float gafMean_1_2[14];
extern const float gfFactor_1_2;
extern const float gafInvCovar_1_2[196];
extern const float gafMean_1_3[14];
extern const float gfFactor_1_3;
extern const float gafInvCovar_1_3[196];
extern const float gafMean_1_4[14];
extern const float gfFactor_1_4;
extern const float gafInvCovar_1_4[196];
extern const float gafMean_1_5[14];
extern const float gfFactor_1_5;
extern const float gafInvCovar_1_5[196];
extern const float gafMean_1_6[14];
extern const float gfFactor_1_6;
extern const float gafInvCovar_1_6[196];
extern const float gafMean_1_7[14];
extern const float gfFactor_1_7;
extern const float gafInvCovar_1_7[196];
extern const float gafWgtList_2[8];
extern const float gafMean_2_0[14];
extern const float gfFactor_2_0;
extern const float gafInvCovar_2_0[196];
extern const float gafMean_2_1[14];
extern const float gfFactor_2_1;
extern const float gafInvCovar_2_1[196];
extern const float gafMean_2_2[14];
extern const float gfFactor_2_2;
extern const float gafInvCovar_2_2[196];
extern const float gafMean_2_3[14];
extern const float gfFactor_2_3;
extern const float gafInvCovar_2_3[196];
extern const float gafMean_2_4[14];
extern const float gfFactor_2_4;
extern const float gafInvCovar_2_4[196];
extern const float gafMean_2_5[14];
extern const float gfFactor_2_5;
extern const float gafInvCovar_2_5[196];
extern const float gafMean_2_6[14];
extern const float gfFactor_2_6;
extern const float gafInvCovar_2_6[196];
extern const float gafMean_2_7[14];
extern const float gfFactor_2_7;
extern const float gafInvCovar_2_7[196];
extern const float gafWgtList_3[8];
extern const float gafMean_3_0[14];
extern const float gfFactor_3_0;
extern const float gafInvCovar_3_0[196];
extern const float gafMean_3_1[14];
extern const float gfFactor_3_1;
extern const float gafInvCovar_3_1[196];
extern const float gafMean_3_2[14];
extern const float gfFactor_3_2;
extern const float gafInvCovar_3_2[196];
extern const float gafMean_3_3[14];
extern const float gfFactor_3_3;
extern const float gafInvCovar_3_3[196];
extern const float gafMean_3_4[14];
extern const float gfFactor_3_4;
extern const float gafInvCovar_3_4[196];
extern const float gafMean_3_5[14];
extern const float gfFactor_3_5;
extern const float gafInvCovar_3_5[196];
extern const float gafMean_3_6[14];
extern const float gfFactor_3_6;
extern const float gafInvCovar_3_6[196];
extern const float gafMean_3_7[14];
extern const float gfFactor_3_7;
extern const float gafInvCovar_3_7[196];
extern const float gafWgtList_4[8];
extern const float gafMean_4_0[14];
extern const float gfFactor_4_0;
extern const float gafInvCovar_4_0[196];
extern const float gafMean_4_1[14];
extern const float gfFactor_4_1;
extern const float gafInvCovar_4_1[196];
extern const float gafMean_4_2[14];
extern const float gfFactor_4_2;
extern const float gafInvCovar_4_2[196];
extern const float gafMean_4_3[14];
extern const float gfFactor_4_3;
extern const float gafInvCovar_4_3[196];
extern const float gafMean_4_4[14];
extern const float gfFactor_4_4;
extern const float gafInvCovar_4_4[196];
extern const float gafMean_4_5[14];
extern const float gfFactor_4_5;
extern const float gafInvCovar_4_5[196];
extern const float gafMean_4_6[14];
extern const float gfFactor_4_6;
extern const float gafInvCovar_4_6[196];
extern const float gafMean_4_7[14];
extern const float gfFactor_4_7;
extern const float gafInvCovar_4_7[196];
extern const float gafWgtList_5[8];
extern const float gafMean_5_0[14];
extern const float gfFactor_5_0;
extern const float gafInvCovar_5_0[196];
extern const float gafMean_5_1[14];
extern const float gfFactor_5_1;
extern const float gafInvCovar_5_1[196];
extern const float gafMean_5_2[14];
extern const float gfFactor_5_2;
extern const float gafInvCovar_5_2[196];
extern const float gafMean_5_3[14];
extern const float gfFactor_5_3;
extern const float gafInvCovar_5_3[196];
extern const float gafMean_5_4[14];
extern const float gfFactor_5_4;
extern const float gafInvCovar_5_4[196];
extern const float gafMean_5_5[14];
extern const float gfFactor_5_5;
extern const float gafInvCovar_5_5[196];
extern const float gafMean_5_6[14];
extern const float gfFactor_5_6;
extern const float gafInvCovar_5_6[196];
extern const float gafMean_5_7[14];
extern const float gfFactor_5_7;
extern const float gafInvCovar_5_7[196];
extern const float gafWgtList_6[8];
extern const float gafMean_6_0[14];
extern const float gfFactor_6_0;
extern const float gafInvCovar_6_0[196];
extern const float gafMean_6_1[14];
extern const float gfFactor_6_1;
extern const float gafInvCovar_6_1[196];
extern const float gafMean_6_2[14];
extern const float gfFactor_6_2;
extern const float gafInvCovar_6_2[196];
extern const float gafMean_6_3[14];
extern const float gfFactor_6_3;
extern const float gafInvCovar_6_3[196];
extern const float gafMean_6_4[14];
extern const float gfFactor_6_4;
extern const float gafInvCovar_6_4[196];
extern const float gafMean_6_5[14];
extern const float gfFactor_6_5;
extern const float gafInvCovar_6_5[196];
extern const float gafMean_6_6[14];
extern const float gfFactor_6_6;
extern const float gafInvCovar_6_6[196];
extern const float gafMean_6_7[14];
extern const float gfFactor_6_7;
extern const float gafInvCovar_6_7[196];
extern const float gafWgtList_7[8];
extern const float gafMean_7_0[14];
extern const float gfFactor_7_0;
extern const float gafInvCovar_7_0[196];
extern const float gafMean_7_1[14];
extern const float gfFactor_7_1;
extern const float gafInvCovar_7_1[196];
extern const float gafMean_7_2[14];
extern const float gfFactor_7_2;
extern const float gafInvCovar_7_2[196];
extern const float gafMean_7_3[14];
extern const float gfFactor_7_3;
extern const float gafInvCovar_7_3[196];
extern const float gafMean_7_4[14];
extern const float gfFactor_7_4;
extern const float gafInvCovar_7_4[196];
extern const float gafMean_7_5[14];
extern const float gfFactor_7_5;
extern const float gafInvCovar_7_5[196];
extern const float gafMean_7_6[14];
extern const float gfFactor_7_6;
extern const float gafInvCovar_7_6[196];
extern const float gafMean_7_7[14];
extern const float gfFactor_7_7;
extern const float gafInvCovar_7_7[196];
extern const float gafWgtList_8[8];
extern const float gafMean_8_0[14];
extern const float gfFactor_8_0;
extern const float gafInvCovar_8_0[196];
extern const float gafMean_8_1[14];
extern const float gfFactor_8_1;
extern const float gafInvCovar_8_1[196];
extern const float gafMean_8_2[14];
extern const float gfFactor_8_2;
extern const float gafInvCovar_8_2[196];
extern const float gafMean_8_3[14];
extern const float gfFactor_8_3;
extern const float gafInvCovar_8_3[196];
extern const float gafMean_8_4[14];
extern const float gfFactor_8_4;
extern const float gafInvCovar_8_4[196];
extern const float gafMean_8_5[14];
extern const float gfFactor_8_5;
extern const float gafInvCovar_8_5[196];
extern const float gafMean_8_6[14];
extern const float gfFactor_8_6;
extern const float gafInvCovar_8_6[196];
extern const float gafMean_8_7[14];
extern const float gfFactor_8_7;
extern const float gafInvCovar_8_7[196];
extern const float gafWgtList_9[8];
extern const float gafMean_9_0[14];
extern const float gfFactor_9_0;
extern const float gafInvCovar_9_0[196];
extern const float gafMean_9_1[14];
extern const float gfFactor_9_1;
extern const float gafInvCovar_9_1[196];
extern const float gafMean_9_2[14];
extern const float gfFactor_9_2;
extern const float gafInvCovar_9_2[196];
extern const float gafMean_9_3[14];
extern const float gfFactor_9_3;
extern const float gafInvCovar_9_3[196];
extern const float gafMean_9_4[14];
extern const float gfFactor_9_4;
extern const float gafInvCovar_9_4[196];
extern const float gafMean_9_5[14];
extern const float gfFactor_9_5;
extern const float gafInvCovar_9_5[196];
extern const float gafMean_9_6[14];
extern const float gfFactor_9_6;
extern const float gafInvCovar_9_6[196];
extern const float gafMean_9_7[14];
extern const float gfFactor_9_7;
extern const float gafInvCovar_9_7[196];
extern const float gafWgtList_10[8];
extern const float gafMean_10_0[14];
extern const float gfFactor_10_0;
extern const float gafInvCovar_10_0[196];
extern const float gafMean_10_1[14];
extern const float gfFactor_10_1;
extern const float gafInvCovar_10_1[196];
extern const float gafMean_10_2[14];
extern const float gfFactor_10_2;
extern const float gafInvCovar_10_2[196];
extern const float gafMean_10_3[14];
extern const float gfFactor_10_3;
extern const float gafInvCovar_10_3[196];
extern const float gafMean_10_4[14];
extern const float gfFactor_10_4;
extern const float gafInvCovar_10_4[196];
extern const float gafMean_10_5[14];
extern const float gfFactor_10_5;
extern const float gafInvCovar_10_5[196];
extern const float gafMean_10_6[14];
extern const float gfFactor_10_6;
extern const float gafInvCovar_10_6[196];
extern const float gafMean_10_7[14];
extern const float gfFactor_10_7;
extern const float gafInvCovar_10_7[196];
extern const float gafWgtList_11[8];
extern const float gafMean_11_0[14];
extern const float gfFactor_11_0;
extern const float gafInvCovar_11_0[196];
extern const float gafMean_11_1[14];
extern const float gfFactor_11_1;
extern const float gafInvCovar_11_1[196];
extern const float gafMean_11_2[14];
extern const float gfFactor_11_2;
extern const float gafInvCovar_11_2[196];
extern const float gafMean_11_3[14];
extern const float gfFactor_11_3;
extern const float gafInvCovar_11_3[196];
extern const float gafMean_11_4[14];
extern const float gfFactor_11_4;
extern const float gafInvCovar_11_4[196];
extern const float gafMean_11_5[14];
extern const float gfFactor_11_5;
extern const float gafInvCovar_11_5[196];
extern const float gafMean_11_6[14];
extern const float gfFactor_11_6;
extern const float gafInvCovar_11_6[196];
extern const float gafMean_11_7[14];
extern const float gfFactor_11_7;
extern const float gafInvCovar_11_7[196];
extern const float gafWgtList_12[8];
extern const float gafMean_12_0[14];
extern const float gfFactor_12_0;
extern const float gafInvCovar_12_0[196];
extern const float gafMean_12_1[14];
extern const float gfFactor_12_1;
extern const float gafInvCovar_12_1[196];
extern const float gafMean_12_2[14];
extern const float gfFactor_12_2;
extern const float gafInvCovar_12_2[196];
extern const float gafMean_12_3[14];
extern const float gfFactor_12_3;
extern const float gafInvCovar_12_3[196];
extern const float gafMean_12_4[14];
extern const float gfFactor_12_4;
extern const float gafInvCovar_12_4[196];
extern const float gafMean_12_5[14];
extern const float gfFactor_12_5;
extern const float gafInvCovar_12_5[196];
extern const float gafMean_12_6[14];
extern const float gfFactor_12_6;
extern const float gafInvCovar_12_6[196];
extern const float gafMean_12_7[14];
extern const float gfFactor_12_7;
extern const float gafInvCovar_12_7[196];
extern const float gafWgtList_13[8];
extern const float gafMean_13_0[14];
extern const float gfFactor_13_0;
extern const float gafInvCovar_13_0[196];
extern const float gafMean_13_1[14];
extern const float gfFactor_13_1;
extern const float gafInvCovar_13_1[196];
extern const float gafMean_13_2[14];
extern const float gfFactor_13_2;
extern const float gafInvCovar_13_2[196];
extern const float gafMean_13_3[14];
extern const float gfFactor_13_3;
extern const float gafInvCovar_13_3[196];
extern const float gafMean_13_4[14];
extern const float gfFactor_13_4;
extern const float gafInvCovar_13_4[196];
extern const float gafMean_13_5[14];
extern const float gfFactor_13_5;
extern const float gafInvCovar_13_5[196];
extern const float gafMean_13_6[14];
extern const float gfFactor_13_6;
extern const float gafInvCovar_13_6[196];
extern const float gafMean_13_7[14];
extern const float gfFactor_13_7;
extern const float gafInvCovar_13_7[196];
extern const float gafWgtList_14[8];
extern const float gafMean_14_0[14];
extern const float gfFactor_14_0;
extern const float gafInvCovar_14_0[196];
extern const float gafMean_14_1[14];
extern const float gfFactor_14_1;
extern const float gafInvCovar_14_1[196];
extern const float gafMean_14_2[14];
extern const float gfFactor_14_2;
extern const float gafInvCovar_14_2[196];
extern const float gafMean_14_3[14];
extern const float gfFactor_14_3;
extern const float gafInvCovar_14_3[196];
extern const float gafMean_14_4[14];
extern const float gfFactor_14_4;
extern const float gafInvCovar_14_4[196];
extern const float gafMean_14_5[14];
extern const float gfFactor_14_5;
extern const float gafInvCovar_14_5[196];
extern const float gafMean_14_6[14];
extern const float gfFactor_14_6;
extern const float gafInvCovar_14_6[196];
extern const float gafMean_14_7[14];
extern const float gfFactor_14_7;
extern const float gafInvCovar_14_7[196];

extern const IMP_LOGNHMMGMM_STORE_S gstLogNHMMGMM_Store;

int IMP_CalcLogProbAll(float *pfVec, int dwVecLen, int dwDim, const IMP_LOGNHMMGMM_STORE_S *pstLNHGS, float *pfProbs, int dwRowLen, int dwColLen);

int IMP_LogNHmmGmm_Create(int dwNeedStateNum, IMP_LOGNHMMGMM_S *pstLogNHmmGmm);
int IMP_LogNHmmGmm_Process(float *pfProbAll, int adwProbAllSize[2], IMP_LOGNHMMGMM_S *pstLogNHmmGmm, int *pdwStateChainOut, int dwChainLen, float *pfScore);
int IMP_LogNHmmGmm_Exit(IMP_LOGNHMMGMM_S *pstLogNHmmGmm);


#endif /* defined(__simpleCNN__lognhmmgmm__) */



