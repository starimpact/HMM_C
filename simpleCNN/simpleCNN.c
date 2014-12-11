//
//  simpleCNN.c
//  simpleCNN
//
//  Created by mzhang on 14/12/8.
//  Copyright (c) 2014年 ___STAR___. All rights reserved.
//


#include "simpleCNN.h"

//the third layer
//dim:32x1
extern float gafParams0_[32];
//dim:1
extern float gafParams1_[1];

//the second layer:
//dim:280x32
extern float gafParams2_[8960];
//dim:32
extern float gafParams3_[32];

//the first layer:
//dim:4x1x5x5
extern float gafParams4_[100];
//dim:4
extern float gafParams5_[4];


int impNormalize(float *pfVec, int dwVecLen)
{
    int dwPI;
    float fMin = pfVec[0], fMax = pfVec[0];
    
    for (dwPI = 0; dwPI < dwVecLen; dwPI++) {
        if (pfVec[dwPI] < fMin) {
            fMin = pfVec[dwPI];
        }
        if (pfVec[dwPI] > fMax) {
            fMax = pfVec[dwPI];
        }
    }
    
    for (dwPI = 0; dwPI < dwVecLen; dwPI++) {
        pfVec[dwPI] = (pfVec[dwPI] - fMin) / (fMax - fMin + 0.000001f);
    }
    
    return 0;
}


float impSigmoid(float fX)
{
    return 1.f / (1 + expf(-fX));
}

int impSpuareConvolute(float *pfKernel, int dwKernelSize, float *pfImageIn, int dwImgWIn, int dwImgHIn, float *pfImageOut, int dwImgWOut, int dwImgHOut)
{
    int dwRI, dwCI;
    int dwBRI, dwBCI, dwBOffset;
    int dwKRI, dwKCI, dwKOffset;
    int dwRadius = (dwKernelSize - 1) >> 1;
    int dwKernLen = dwKernelSize * dwKernelSize;
    float fVal = 0.f;
    
    if (dwImgHIn - dwKernelSize + 1 != dwImgHOut || dwImgWIn - dwKernelSize + 1 != dwImgWOut) {
        printf("error ...[%d, %d] [%d, %d]\n", dwImgHIn - dwKernelSize + 1, dwImgHOut, dwImgWIn - dwKernelSize + 1, dwImgWOut);
        return 0;
    }
    
    for (dwRI = dwRadius; dwRI < dwImgHIn - dwRadius; dwRI++) {
        for (dwCI = dwRadius; dwCI < dwImgWIn - dwRadius; dwCI++) {
            fVal = 0.f;
            for (dwBRI = dwRI - dwRadius, dwKRI = 0; dwBRI <= dwRI + dwRadius; dwBRI++, dwKRI++) {
                for (dwBCI = dwCI - dwRadius, dwKCI = 0; dwBCI <= dwCI + dwRadius; dwBCI++, dwKCI++) {
                    dwBOffset = dwBRI * dwImgWIn + dwBCI;
                    dwKOffset = dwKRI * dwKernelSize + dwKCI;
                    fVal += pfImageIn[dwBOffset] * pfKernel[dwKernLen - dwKOffset - 1];
                }
            }
            
            pfImageOut[(dwRI - dwRadius) * dwImgWOut + (dwCI - dwRadius)] += fVal;
        }
    }
    
    
    return 0;
}


int impMaxPooling(float *pfImgIn, int dwImgInW, int dwImgInH, int dwPoolingSize, float *pfImgOut, int dwImgOutW, int dwImgOutH)
{
    int dwRI, dwCI;
    int dwBRI, dwBCI;
    int dwWRI = 0, dwWCI = 0;
    int dwBOffset;
    float fMaxVal = 0.f;
    
    if (dwImgInH != dwImgOutH * dwPoolingSize || dwImgInW != dwImgOutW * dwPoolingSize) {
        printf("error...[%d, %d] [%d, %d]\n", dwImgInH, dwImgOutH * dwPoolingSize, dwImgInW, dwImgOutW * dwPoolingSize);
        return 0;
    }
    
    for (dwRI = 0, dwWRI = 0; dwRI <= dwImgInH - dwPoolingSize; dwRI += dwPoolingSize, dwWRI++) {
        for (dwCI = 0, dwWCI = 0; dwCI <= dwImgInW - dwPoolingSize; dwCI += dwPoolingSize, dwWCI++) {
            fMaxVal = pfImgIn[dwRI * dwImgInW + dwCI];
            for (dwBRI = dwRI; dwBRI < dwRI + dwPoolingSize; dwBRI++) {
                for (dwBCI = dwCI; dwBCI < dwCI + dwPoolingSize; dwBCI++) {
                    dwBOffset = dwBRI * dwImgInW + dwBCI;
                    if (pfImgIn[dwBOffset] > fMaxVal) {
                        fMaxVal = pfImgIn[dwBOffset];
                    }
                }
            }
            pfImgOut[dwWRI * dwImgOutW + dwWCI] = fMaxVal;
        }
    }
//    printf("[%d, %d] [%d, %d]\n", dwImgOutW, dwImgOutH, dwWCI, dwWRI);
    return 0;
}


int impCovLayer_Create(int dwImgNum_In, int adwImgSize_In[2], int dwKernelSize, int dwPoolingSize, int dwImgNum_Out, float *pfWeight, float *pfBias, IMP_COVLAYER_S *pstCovLayer)
{
    pstCovLayer->dwKernelSize = dwKernelSize;
    pstCovLayer->dwPoolingSize = dwPoolingSize;
    pstCovLayer->dwImgNum_In = dwImgNum_In;
    memcpy(pstCovLayer->adwImgSize_In, adwImgSize_In, sizeof(int) * 2);
    pstCovLayer->dwImgNum_Out = dwImgNum_Out;
    
    pstCovLayer->adwCovImageSize[0] = adwImgSize_In[0] - dwKernelSize + 1;
    pstCovLayer->adwCovImageSize[1] = adwImgSize_In[1] - dwKernelSize + 1;
    pstCovLayer->pfCovResult = (float*)malloc(dwImgNum_Out * pstCovLayer->adwCovImageSize[0] * pstCovLayer->adwCovImageSize[1] * sizeof(float));
    
    pstCovLayer->adwImgSize_Out[0] = pstCovLayer->adwCovImageSize[0] / dwPoolingSize;
    pstCovLayer->adwImgSize_Out[1] = pstCovLayer->adwCovImageSize[1] / dwPoolingSize;
    pstCovLayer->pfImages_Out = (float*)malloc(dwImgNum_Out * pstCovLayer->adwImgSize_Out[0] * pstCovLayer->adwImgSize_Out[1] * sizeof(float));
    
    pstCovLayer->pfWeight = (float*)malloc(dwImgNum_Out * dwImgNum_In * dwKernelSize * dwKernelSize * sizeof(float));
    pstCovLayer->pfBias = (float*)malloc(dwImgNum_Out * sizeof(float));
    
    memcpy(pstCovLayer->pfWeight, pfWeight, dwImgNum_Out * dwImgNum_In * dwKernelSize * dwKernelSize * sizeof(float));
    memcpy(pstCovLayer->pfBias, pfBias, dwImgNum_Out * sizeof(float));
    
    return 0;
}


int impCovLayer_Process(float *pfImages_In, IMP_COVLAYER_S *pstCovLayer)
{
    int dwI;
    int dwImgInI, dwImgOutI;
    int dwImgNum_In = 0, dwImgNum_Out = 0, adwImgSize_In[2], adwCovImageSize[2], adwImgSize_Out[2];
    int dwPoolingSize = 0, dwKernelSize = 0;
    float *pfCovResult = 0, *pfImages_Out = 0, *pfWeight = 0, *pfBias = 0;
    float *pfImageInOne = 0, *pfImageOutOne = 0, *pfImageCovOne = 0;
    float *pfKernelRow = 0, *pfKernelOne = 0;
    int dwImgInSize = 0, dwImgCovSize = 0, dwImgOutSize = 0, dwKernelSquareSize;
    
    dwImgNum_In = pstCovLayer->dwImgNum_In;
    dwImgNum_Out = pstCovLayer->dwImgNum_Out;
    pfImages_Out = pstCovLayer->pfImages_Out;
    pfCovResult = pstCovLayer->pfCovResult;
    pfWeight = pstCovLayer->pfWeight;
    pfBias = pstCovLayer->pfBias;
    
    adwImgSize_In[0] = pstCovLayer->adwImgSize_In[0];
    adwImgSize_In[1] = pstCovLayer->adwImgSize_In[1];
    dwImgInSize = adwImgSize_In[0] * adwImgSize_In[1];
    
    adwImgSize_Out[0] = pstCovLayer->adwImgSize_Out[0];
    adwImgSize_Out[1] = pstCovLayer->adwImgSize_Out[1];
    dwImgOutSize = adwImgSize_Out[0] * adwImgSize_Out[1];
    
    adwCovImageSize[0] = pstCovLayer->adwCovImageSize[0];
    adwCovImageSize[1] = pstCovLayer->adwCovImageSize[1];
    dwImgCovSize = adwCovImageSize[0] * adwCovImageSize[1];
    
    dwPoolingSize = pstCovLayer->dwPoolingSize;
    dwKernelSize = pstCovLayer->dwKernelSize;
    dwKernelSquareSize = dwKernelSize * dwKernelSize;

#if SCNN_DBG
    {
        int dwI, dwRI, dwCI;
        
        printf("impCovLayer_Process_pfImages_In:\n");
//        for (dwI = 0; dwI < dwImgInSize; dwI++) {
//            printf("%f, ", pfImages_In[dwI]);
 //       }
        for (dwRI = 0; dwRI < 5; dwRI++) {
            for (dwCI = 0; dwCI < 5; dwCI++) {
                printf("%f, ", pfImages_In[dwRI * adwImgSize_In[1] + dwCI]);
            }
            printf("\n");
        }
        printf("\n");
    }
#endif
    
    for (dwImgOutI = 0; dwImgOutI < dwImgNum_Out; dwImgOutI++) {
        pfImageCovOne = pfCovResult + dwImgOutI * dwImgCovSize;
        memset(pfImageCovOne, 0, dwImgCovSize * sizeof(float));
        pfKernelRow = pfWeight + dwImgOutI * dwImgNum_In * dwKernelSquareSize;
        for (dwImgInI = 0; dwImgInI < dwImgNum_In; dwImgInI++) {
            pfImageInOne = pfImages_In + dwImgInI * dwImgInSize;
            pfKernelOne = pfKernelRow + dwImgInI * dwKernelSquareSize;
#if SCNN_DBG
            {
                int dwI;
                
                printf("impCovLayer_Process_pfKernelOne:\n");
                for (dwI = 0; dwI < dwKernelSquareSize; dwI++) {
                    printf("%f, ", pfKernelOne[dwI]);
                }
                printf("\n");
            }
#endif
            impSpuareConvolute(pfKernelOne, dwKernelSize, pfImageInOne, adwImgSize_In[1], adwImgSize_In[0], pfImageCovOne, adwCovImageSize[1], adwCovImageSize[0]);
        }
        
#if SCNN_DBG
        {
            int dwI;
            
            printf("impCovLayer_Process_pfImageCovOne:\n");
            for (dwI = 0; dwI < dwImgCovSize; dwI++) {
                printf("%f, ", pfImageCovOne[dwI]);
            }
            printf("\n");
        }
#endif
        
        pfImageOutOne = pfImages_Out + dwImgOutI * dwImgOutSize;
        impMaxPooling(pfImageCovOne, adwCovImageSize[1], adwCovImageSize[0], dwPoolingSize, pfImageOutOne, adwImgSize_Out[1], adwImgSize_Out[0]);
#if SCNN_DBG
        {
            int dwI;
            
            printf("impCovLayer_Process_pfImageOutOne_maxpooling:\n");
            for (dwI = 0; dwI < dwImgOutSize; dwI++) {
                printf("%f, ", pfImageOutOne[dwI]);
            }
            printf("\n");
        }
#endif
        for (dwI = 0; dwI < dwImgOutSize; dwI++) {
            pfImageOutOne[dwI] = impSigmoid(pfImageOutOne[dwI] + pfBias[dwImgOutI]);
        }
#if SCNN_DBG
        {
            int dwI;
            
            printf("impCovLayer_Process_pfImageOutOne_sigmoid:\n");
            for (dwI = 0; dwI < dwImgOutSize; dwI++) {
                printf("%f, ", pfImageOutOne[dwI]);
            }
            printf("\n");
        }
#endif
    //    exit(1);
    //    break;
    }
    
    
    return 0;
}


int impCovLayer_Exit(IMP_COVLAYER_S *pstCovLayer)
{
    free(pstCovLayer->pfCovResult);
    free(pstCovLayer->pfImages_Out);
    free(pstCovLayer->pfWeight);
    free(pstCovLayer->pfBias);
    
    return 0;
}


int impHiddenLayer_Create(int dwNodeInputLen, int dwNodeOutputLen, float *pfWeight, float *pfBias, IMP_HIDDENLAYER_S *pstHiddenLayer)
{
    pstHiddenLayer->dwNodeInputLen = dwNodeInputLen;
    pstHiddenLayer->dwNodeOutputLen = dwNodeOutputLen;
    
    pstHiddenLayer->pfNodeOutput = (float*)malloc(dwNodeOutputLen * sizeof(float));
    pstHiddenLayer->pfBias = (float*)malloc(dwNodeOutputLen * sizeof(float));
    pstHiddenLayer->pfWeight = (float*)malloc(dwNodeOutputLen * dwNodeInputLen * sizeof(float));
    
    memcpy(pstHiddenLayer->pfWeight, pfWeight, dwNodeOutputLen * dwNodeInputLen * sizeof(float));
    memcpy(pstHiddenLayer->pfBias, pfBias, dwNodeOutputLen * sizeof(float));
    
    return 0;
}


int impHiddenLayer_Process(float *pfNodeInput, IMP_HIDDENLAYER_S *pstHiddenLayer)
{
    int dwNInI, dwNOutI;
    float *pfWeight = 0, *pfBias = 0, *pfNodeOutput = 0;
    float *pfWeightRow = 0;
    int dwNodeOutputLen, dwNodeInputLen;
    
    dwNodeInputLen = pstHiddenLayer->dwNodeInputLen;
    dwNodeOutputLen = pstHiddenLayer->dwNodeOutputLen;
    pfNodeOutput = pstHiddenLayer->pfNodeOutput;
    pfWeight = pstHiddenLayer->pfWeight;
    pfBias = pstHiddenLayer->pfBias;
#if SCNN_DBG
    {
        int dwI;
        
        printf("impHiddenLayer_Process_pfNodeInput:\n");
        for (dwI = 0; dwI < dwNodeInputLen; dwI++) {
            printf("%f, ", pfNodeInput[dwI]);
        }
        printf("\n");
    }
#endif
    for (dwNOutI = 0; dwNOutI < dwNodeOutputLen; dwNOutI++) {
        pfWeightRow = pfWeight + dwNOutI * dwNodeInputLen;
        pfNodeOutput[dwNOutI] = 0;
        for (dwNInI = 0; dwNInI < dwNodeInputLen; dwNInI++) {
            pfNodeOutput[dwNOutI] += pfWeightRow[dwNInI] * pfNodeInput[dwNInI];
        }
     //   pfNodeOutput[dwNOutI] = (pfNodeOutput[dwNOutI] + pfBias[dwNOutI]);
        pfNodeOutput[dwNOutI] = impSigmoid(pfNodeOutput[dwNOutI] + pfBias[dwNOutI]);
    }
#if SCNN_DBG
    {
        int dwI;
        
        printf("impHiddenLayer_Process_pfNodeOutput:\n");
        for (dwI = 0; dwI < dwNodeOutputLen; dwI++) {
            printf("%f, ", pfNodeOutput[dwI]);
        }
        printf("\n");
    }
#endif
//    exit(2);
    return 0;
}


int impHiddenLayer_Exit(IMP_HIDDENLAYER_S *pstHiddenLayer)
{
    free(pstHiddenLayer->pfBias);
    free(pstHiddenLayer->pfWeight);
    free(pstHiddenLayer->pfNodeOutput);
    
    return 0;
}


int impLogRegLayer_Create(int dwNodeInputLen, int dwNodeOutputLen, float *pfWeight, float *pfBias, IMP_LOGREGLAYER_S *pstLogRegLayer)
{
    pstLogRegLayer->dwNodeInputLen = dwNodeInputLen;
    pstLogRegLayer->dwNodeOutputLen = dwNodeOutputLen;
    
    pstLogRegLayer->pfNodeOutput = (float*)malloc(dwNodeOutputLen * sizeof(float));
    pstLogRegLayer->pfBias = (float*)malloc(dwNodeOutputLen * sizeof(float));
    pstLogRegLayer->pfWeight = (float*)malloc(dwNodeOutputLen * dwNodeInputLen * sizeof(float));
    
    memcpy(pstLogRegLayer->pfWeight, pfWeight, dwNodeOutputLen * dwNodeInputLen * sizeof(float));
    memcpy(pstLogRegLayer->pfBias, pfBias, dwNodeOutputLen * sizeof(float));
    
    return 0;
}


int impLogRegLayer_Process(float *pfNodeInput, IMP_LOGREGLAYER_S *pstLogRegLayer)
{
    int dwNInI, dwNOutI;
    float *pfWeight = 0, *pfBias = 0, *pfNodeOutput = 0;
    float *pfWeightRow = 0;
    int dwNodeOutputLen, dwNodeInputLen;
    
    dwNodeInputLen = pstLogRegLayer->dwNodeInputLen;
    dwNodeOutputLen = pstLogRegLayer->dwNodeOutputLen;
    pfNodeOutput = pstLogRegLayer->pfNodeOutput;
    pfWeight = pstLogRegLayer->pfWeight;
    pfBias = pstLogRegLayer->pfBias;
    
    for (dwNOutI = 0; dwNOutI < dwNodeOutputLen; dwNOutI++) {
        pfWeightRow = pfWeight + dwNOutI * dwNodeInputLen;
        pfNodeOutput[dwNOutI] = 0;
        for (dwNInI = 0; dwNInI < dwNodeInputLen; dwNInI++) {
            pfNodeOutput[dwNOutI] += pfWeightRow[dwNInI] * pfNodeInput[dwNInI];
        }
        pfNodeOutput[dwNOutI] = impSigmoid(pfNodeOutput[dwNOutI] + pfBias[dwNOutI]);
    }
    
#if SCNN_DBG
    {
        int dwI;
        
        printf("mpLogRegLayer_Process_pfNodeOutput:\n");
        for (dwI = 0; dwI < dwNodeOutputLen; dwI++) {
            printf("%f, ", pfNodeOutput[dwI]);
        }
        printf("\n");
    }
#endif
    
    return 0;
}


int impLogRegLayer_Exit(IMP_LOGREGLAYER_S *pstLogRegLayer)
{
    free(pstLogRegLayer->pfBias);
    free(pstLogRegLayer->pfWeight);
    free(pstLogRegLayer->pfNodeOutput);
    
    return 0;
}




int IMP_SimpleCNN_Create(int dwImgW, int dwImgH, IMP_SIMPLECNN_S *pstSimpleCNN)
{
    int adwImgSize_In[2];
    int dwNodeInputLen;
    
    adwImgSize_In[0] = dwImgH;
    adwImgSize_In[1] = dwImgW;
    pstSimpleCNN->pfImage = (float*)malloc(dwImgW * dwImgH * sizeof(float));
    pstSimpleCNN->adwImageSize[0] = adwImgSize_In[0];
    pstSimpleCNN->adwImageSize[1] = adwImgSize_In[1];
    impCovLayer_Create(1, adwImgSize_In, 5, 2, 4, gafParams4_, gafParams5_, &pstSimpleCNN->stCovLayer);
    dwNodeInputLen = pstSimpleCNN->stCovLayer.dwImgNum_Out * pstSimpleCNN->stCovLayer.adwImgSize_Out[0] * pstSimpleCNN->stCovLayer.adwImgSize_Out[1];
    impHiddenLayer_Create(dwNodeInputLen, 32, gafParams2_, gafParams3_, &pstSimpleCNN->stHiddenLayer);
    impLogRegLayer_Create(pstSimpleCNN->stHiddenLayer.dwNodeOutputLen, 1, gafParams0_, gafParams1_, &pstSimpleCNN->stLogRegLayer);
    
    return 0;
}


int IMP_SimpleCNN_Process(uchar *pubyImage, IMP_SIMPLECNN_S *pstSimpleCNN)
{
    int dwPI;
    int dwImgSize;
    float *pfImage = 0;
    int dwScore;
    
    dwImgSize = pstSimpleCNN->adwImageSize[0] * pstSimpleCNN->adwImageSize[1];
    pfImage = pstSimpleCNN->pfImage;
    for (dwPI = 0; dwPI < dwImgSize; dwPI++) {
        pfImage[dwPI] = pubyImage[dwPI];
    }
    
//    printf("%f\n", pfImage[10]);
    impNormalize(pfImage, dwImgSize);
//    printf("%f\n", pfImage[10]);
    
    impCovLayer_Process(pfImage, &pstSimpleCNN->stCovLayer);
//    exit(1);
    impHiddenLayer_Process(pstSimpleCNN->stCovLayer.pfImages_Out, &pstSimpleCNN->stHiddenLayer);
    
    impLogRegLayer_Process(pstSimpleCNN->stHiddenLayer.pfNodeOutput, &pstSimpleCNN->stLogRegLayer);
    
    dwScore = pstSimpleCNN->stLogRegLayer.pfNodeOutput[0] * 255;
    
    return dwScore;
}


int IMP_SimpleCNN_Exit(IMP_SIMPLECNN_S *pstSimpleCNN)
{
    free(pstSimpleCNN->pfImage);
    impCovLayer_Exit(&pstSimpleCNN->stCovLayer);
    impHiddenLayer_Exit(&pstSimpleCNN->stHiddenLayer);
    impLogRegLayer_Exit(&pstSimpleCNN->stLogRegLayer);
    return 0;
}



