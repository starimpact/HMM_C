//
//  lnhgLPCE.c
//  simpleCNN
//
//  Created by mzhang on 14/12/11.
//  Copyright (c) 2014年 ___STAR___. All rights reserved.
//

#include "lnhgLPCE.h"
#include "lognhmmgmm.h"
#include "simpleCNN.h"

#include "cv.h"
#include "highgui.h"


void imgResize(uchar *patch, int s32W_src, int s32H_src, uchar *result, int s32W_dst, int s32H_dst)
{
    int s32RI, s32CI;
    float srcX, srcY;
    float fSub_X, fSub_Y;
    int s32x, s32y;
    float fCov_X = 1.0 * (s32W_src - 1) / (s32W_dst - 1);
    float fCov_Y = 1.0 * (s32H_src - 1) / (s32H_dst - 1);
    float fTool = 0;
    
    for(s32RI = 0; s32RI < s32H_dst; s32RI++)
    {
        srcY = s32RI * fCov_Y;
        s32y = (int)srcY;
        fSub_Y = srcY - s32y;
        uchar *ptrPatch = patch + s32y * s32W_src;
        uchar *ptrResult = result + s32RI * s32W_dst;
        for(s32CI=0; s32CI < s32W_dst; s32CI++)
        {
            srcX = s32CI * fCov_X;
            s32x = (int)srcX;
            fSub_X = srcX - s32x;
            fTool = fSub_X * fSub_Y;
            if((s32x == s32W_src - 1) || (s32y == s32H_src - 1))
                ptrResult[s32CI] = ptrPatch[s32x];
            else
                ptrResult[s32CI] = (uchar)((1 - fSub_X - fSub_Y + fTool) * ptrPatch[s32x] +
                                            (fSub_Y - fTool) * ptrPatch[s32x + s32W_src] + (fSub_X - fTool) * ptrPatch[s32x + 1] +
                                            fTool * ptrPatch[s32x + s32W_src + 1] + 0.5);
        }
    }
}


int IMP_LNHG_LPCE_Create()
{
    return 0;
}


int IMP_LNHG_LPCE_Process(uchar *pubyImg, int dwImgW, int dwImgH)
{
    int dwSI, dwRI, dwCI;
    IMP_SIMPLECNN_S stSCNN;
    IMP_LOGNHMMGMM_S stLogNHMMGMM;
    float *pfProbAll = 0;
    int adwProbAllSize[2], adwStateNum[3] = {13, 14, 15};
    int dwW, dwH;
    float fTime1, fTime2, fFrequency;
    float afScores[1024], fBestScore, fNowScore;
    int adwStateChain[1024], adwStateChainBest[1024], dwBestWidth = 0;
    uchar *pubyImgCNN = 0, *pubyImgStd = 0;
    int dwImgStdW, dwImgStdH;
    int dwScore = 0;
    
    
    dwImgStdH = WIN_H;
    dwImgStdW = dwImgW * dwImgStdH / dwImgH;
    dwW = dwImgStdW;
    dwH = dwImgStdH;
    
    adwProbAllSize[0] = STATE_NUM;
    adwProbAllSize[1] = dwW - DIM_NUM;
    pfProbAll = (float*)malloc(adwProbAllSize[0] * adwProbAllSize[1] * sizeof(float));
    
    pubyImgStd = (uchar*)malloc(dwImgStdW * dwImgStdH);
    
    imgResize(pubyImg + 4 * dwImgW, dwImgW, dwImgH - 4 * 2, pubyImgStd, dwImgStdW, dwImgStdH);
    
    printf("stdimgsize:%d, %d\n", dwW, dwH);
    
    pubyImgCNN = (uchar*)malloc(WIN_W * WIN_H);
    
    IMP_SimpleCNN_Create(WIN_W, WIN_H, &stSCNN);
    
    fFrequency = cvGetTickFrequency();
    
    fTime1 = cvGetTickCount();
    memset(afScores, 0, dwW * 4);
    for (dwSI = 0; dwSI < dwW - WIN_W; dwSI++) {
        for (dwRI = 0; dwRI < dwH; dwRI++) {
            memcpy(pubyImgCNN + dwRI * WIN_W, pubyImgStd + dwRI * dwImgStdW + dwSI, WIN_W);
        }
        dwScore = IMP_SimpleCNN_Process(pubyImgCNN, &stSCNN);
        afScores[dwSI + WIN_W / 2] = dwScore;
    }
#if 0
    for (dwSI = 0; dwSI < dwW; dwSI++) {
        printf("%.1f, ", afScores[dwSI]);
    }
    printf("\n");
#endif
    IMP_CalcLogProbAll(afScores, dwW, DIM_NUM, &gstLogNHMMGMM_Store, pfProbAll, adwProbAllSize[0], adwProbAllSize[1]);
    fBestScore = -1e10;
    for (dwSI = 0; dwSI < 3; dwSI++) {
        IMP_LogNHmmGmm_Create(adwStateNum[dwSI], &stLogNHMMGMM);
        memset(adwStateChain, 0, sizeof(int) * dwW);
        IMP_LogNHmmGmm_Process(pfProbAll, adwProbAllSize, &stLogNHMMGMM, adwStateChain + DIM_NUM/2, dwW, &fNowScore);
        
        if (fBestScore < fNowScore) {
            dwBestWidth = adwStateNum[dwSI];
            fBestScore = fNowScore;
            memcpy(adwStateChainBest, adwStateChain, dwW * sizeof(int));
        }
#if 0
        printf("statenum:%d, score:%.5f\n", dwSI, fNowScore);
        printf("state chain:");
        for (dwRI = 0; dwRI < dwW; dwRI++) {
            printf("%d-", adwStateChain[dwRI]);
        }
        printf("\n");
#endif
        IMP_LogNHmmGmm_Exit(&stLogNHMMGMM);
    }
    printf("width:%d, score:%.1f\n", dwBestWidth, fBestScore);
    
    
    fTime2 = cvGetTickCount();
    
    printf("cost:%f ms\n", (fTime2 - fTime1) / fFrequency / 1000);
    
    {
        IplImage *pstImgColor = cvCreateImage(cvSize(dwW, dwH), 8, 3);
        
        for (dwRI = 0; dwRI < dwH; dwRI++) {
            for (dwCI = 0; dwCI < dwW; dwCI++) {
                pstImgColor->imageData[dwRI * pstImgColor->widthStep + dwCI * 3 + 0] = pubyImgStd[dwRI * dwW + dwCI];
                pstImgColor->imageData[dwRI * pstImgColor->widthStep + dwCI * 3 + 1] = pubyImgStd[dwRI * dwW + dwCI];
                pstImgColor->imageData[dwRI * pstImgColor->widthStep + dwCI * 3 + 2] = pubyImgStd[dwRI * dwW + dwCI];
            }
        }
        for (dwSI = 0; dwSI < dwW - 1; dwSI++) {
            if (adwStateChainBest[dwSI] == 1 || (adwStateChainBest[dwSI] > 0 && adwStateChainBest[dwSI + 1] == 0)) {
                cvLine(pstImgColor, cvPoint(dwSI, 0), cvPoint(dwSI, dwH-1), CV_RGB(255, 0, 0), 1, 0, 0);
            }
        }
        
        cvShowImage("segment", pstImgColor);
        cvWaitKey(40);
        cvReleaseImage(&pstImgColor);
    }
    
    IMP_SimpleCNN_Exit(&stSCNN);
    
    free(pfProbAll);
    free(pubyImgStd);
    free(pubyImgCNN);
    
    return 0;
}


int IMP_LNHG_LPCE_Exit()
{
    return 0;
}

