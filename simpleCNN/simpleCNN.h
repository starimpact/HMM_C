//
//  simpleCNN.h
//  simpleCNN
//
//  Created by mzhang on 14/12/8.
//  Copyright (c) 2014年 ___STAR___. All rights reserved.
//

#ifndef __simpleCNN__simpleCNN__
#define __simpleCNN__simpleCNN__

#include <stdio.h>
#include "stdlib.h"
#include "string.h"
#include "math.h"

#define SCNN_DBG 0

typedef unsigned char uchar;

typedef struct impCovLayer_S
{
    int adwImgSize_In[2];
    int dwImgNum_In;
    int dwKernelSize;
    int dwPoolingSize;
    
    float *pfCovResult;
    int adwCovImageSize[2];
    
    float *pfImages_Out;
    int adwImgSize_Out[2];
    int dwImgNum_Out;
    
    float *pfWeight;
    float *pfBias;
} IMP_COVLAYER_S;


typedef struct impHiddenLayer_S
{
    int dwNodeInputLen;
    int dwNodeOutputLen;
    float *pfNodeOutput;
    
    float *pfWeight;
    float *pfBias;
} IMP_HIDDENLAYER_S;


typedef struct impLogRegLayer_S
{
    int dwNodeInputLen;
    int dwNodeOutputLen;
    float *pfNodeOutput;
    
    float *pfWeight;
    float *pfBias;
    
} IMP_LOGREGLAYER_S;


//input shape: 32x14
typedef struct impSimpleCNN_S
{
    float *pfImage;
    int adwImageSize[2];
    
    IMP_COVLAYER_S stCovLayer;
    IMP_HIDDENLAYER_S stHiddenLayer;
    IMP_LOGREGLAYER_S stLogRegLayer;
} IMP_SIMPLECNN_S;

int impCovLayer_Create(int dwImgNum_In, int adwImgSize_In[2], int dwKernelSize, int dwPoolingSize, int dwImgNum_Out, float *pfWeight, float *pfBias, IMP_COVLAYER_S *pstCovLayer);
int impCovLayer_Process(float *pfImages_In, IMP_COVLAYER_S *pstCovLayer);
int impCovLayer_Exit(IMP_COVLAYER_S *pstCovLayer);

int impHiddenLayer_Create(int dwNodeInputLen, int dwNodeOutputLen, float *pfWeight, float *pfBias, IMP_HIDDENLAYER_S *pstHiddenLayer);
int impHiddenLayer_Process(float *pfNodeInput, IMP_HIDDENLAYER_S *pstHiddenLayer);
int impHiddenLayer_Exit(IMP_HIDDENLAYER_S *pstHiddenLayer);

int impLogRegLayer_Create(int dwNodeInputLen, int dwNodeOutputLen, float *pfWeight, float *pfBias, IMP_LOGREGLAYER_S *pstLogRegLayer);
int impLogRegLayer_Process(float *pfNodeInput, IMP_LOGREGLAYER_S *pstLogRegLayer);
int impLogRegLayer_Exit(IMP_LOGREGLAYER_S *pstLogRegLayer);




int IMP_SimpleCNN_Create(int dwImgW, int dwImgH, IMP_SIMPLECNN_S *pstSimpleCNN);
int IMP_SimpleCNN_Process(uchar *pubyImage, IMP_SIMPLECNN_S *pstSimpleCNN);
int IMP_SimpleCNN_Exit(IMP_SIMPLECNN_S *pstSimpleCNN);


#endif /* defined(__simpleCNN__simpleCNN__) */







