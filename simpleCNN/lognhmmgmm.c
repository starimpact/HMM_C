//
//  lognhmmgmm.c
//  simpleCNN
//
//  Created by mzhang on 14/12/10.
//  Copyright (c) 2014年 ___STAR___. All rights reserved.
//

#include "lognhmmgmm.h"

float imp_CalcGSMLogProb(float *pfX, int dwDim, const IMP_LOGGSM_S *pstLGSM)
{
    int dwRI, dwCI;
    const float fFactor = pstLGSM->fFactor;
    const float *pfMean = pstLGSM->pfMean;
    const float *pfInvCovar = pstLGSM->pfInvCovar;
    float fLogGSMProb = 0, fTmp = 0;
    
    for (dwRI = 0; dwRI < dwDim; dwRI++) {
        fTmp = 0.f;
        for (dwCI = 0; dwCI < dwDim; dwCI++) {
            fTmp += (pfX[dwCI] - pfMean[dwCI]) * pfInvCovar[dwRI * dwDim + dwCI];
        }
        fLogGSMProb += fTmp * (pfX[dwRI] - pfMean[dwRI]);
    }
    
    fLogGSMProb = -fLogGSMProb / 2 + fFactor;
    
    
    return fLogGSMProb;
}


int imp_CalcPostProbaByLog(float *pfX, int dwDim, const IMP_LOGGMM_S *pstLGMM, int dwGMMSize, float *pfPostAll, float *pfLogProbList, float *pfLogWgtList)
{
    int dwI, dwI2;
    float fTmp1, fMax, fTmp2;
    float *pfProbWgt = 0, *pfSub = 0;
    
    
    pfProbWgt = (float*)malloc(dwGMMSize * sizeof(float));
    pfSub = (float*)malloc(dwGMMSize * sizeof(float));
    
    for (dwI = 0; dwI < dwGMMSize; dwI++) {
        pfLogProbList[dwI] = imp_CalcGSMLogProb(pfX, dwDim, &pstLGMM->astLogGSM[dwI]);
        pfLogWgtList[dwI] = logf(pstLGMM->pfWeights[dwI]);
        pfProbWgt[dwI] = pfLogProbList[dwI] + pfLogWgtList[dwI];
    }
    
    for (dwI = 0; dwI < dwGMMSize; dwI++) {
        fMax = 0;
        for (dwI2 = 0; dwI2 < dwGMMSize; dwI2++) {
            pfSub[dwI2] = pfProbWgt[dwI2] - pfProbWgt[dwI];
            if (fMax < pfSub[dwI2]) {
                fMax = pfSub[dwI2];
            }
        }
        fTmp2 = MINIMUM_V;
        if (fMax < 64) {
            fTmp1 = 0;
            for (dwI2 = 0; dwI2 < dwGMMSize; dwI2++) {
                fTmp1 += expf(pfSub[dwI2]);
            }
            fTmp2 = 1 / fTmp1;
        }
        pfPostAll[dwI] = fTmp2;
    }
    
    
    free(pfSub);
    free(pfProbWgt);
    
    return 0;
}


float imp_CalcGMMLogProb(float *pfX, int dwDim, const IMP_LOGGMM_S *pstLGMM, int dwGMMSize)
{
    int dwI;
    float fLogGMMProb = 0.f;
    float *pfPostAll, *pfLogProbList, *pfLogWgtList;
    
    pfPostAll = (float*)malloc(dwGMMSize * sizeof(float));
    pfLogProbList = (float*)malloc(dwGMMSize * sizeof(float));
    pfLogWgtList = (float*)malloc(dwGMMSize * sizeof(float));
    
    imp_CalcPostProbaByLog(pfX, dwDim, pstLGMM, dwGMMSize, pfPostAll, pfLogProbList, pfLogWgtList);
    
    fLogGMMProb = 0;
    for (dwI = 0; dwI < dwGMMSize; dwI++) {
        fLogGMMProb += pfPostAll[dwI] * (pfLogProbList[dwI] + pfLogWgtList[dwI] - logf(pfPostAll[dwI]));
    }
    
    free(pfPostAll);
    free(pfLogProbList);
    free(pfLogWgtList);
    
    return fLogGMMProb;
}


int IMP_CalcLogProbAll(float *pfVec, int dwVecLen, int dwDim, const IMP_LOGNHMMGMM_STORE_S *pstLNHGS, float *pfProbs, int dwRowLen, int dwColLen)
{
    int dwPI, dwI;
    float *pfX = 0;
    
    if (dwDim != DIM_NUM || dwRowLen != STATE_NUM || dwColLen != dwVecLen - dwDim) {
        printf("error....\n");
    }
    
    for (dwPI = 0; dwPI < dwVecLen - dwDim; dwPI++) {
        pfX = pfVec + dwPI;
        for (dwI = 0; dwI < dwRowLen; dwI++) {
            pfProbs[dwI * dwColLen + dwPI] = imp_CalcGMMLogProb(pfX, dwDim, &pstLNHGS->astLogGMMList[dwI], GMM_SIZE);
        }
    }
    
    
    return 0;
}


int imp_doViterbi(float *pfProbAll, int dwProbRow, int dwProbCol, float *pfPrior, float *pfTrans, int *pdwStateChainOut, int dwChainLen, float *pfScore)
{
    int dwRI, dwRI2, dwCI;
    float *pfProbCol1 = 0, *pfProbCol2 = 0;
    int *pdwPathNet = 0;
    float fMaxScore, fTmp;
    int dwMaxIdx;
    
    pfProbCol1 = (float*)malloc(dwProbRow * sizeof(float));
    pfProbCol2 = (float*)malloc(dwProbRow * sizeof(float));
    pdwPathNet = (int*)malloc(dwProbRow * dwProbCol * sizeof(int));
    
    for (dwRI = 0; dwRI < dwProbRow; dwRI++) {
        pfProbCol1[dwRI] = pfProbAll[dwRI * dwProbCol] + pfPrior[dwRI];
    }
    for (dwCI = 0; dwCI < dwProbCol; dwCI++) {
        for (dwRI = 0; dwRI < dwProbRow; dwRI++) {
            fTmp = 0;
            fMaxScore = pfProbCol1[0] + pfTrans[0 * dwProbRow + dwRI] + pfProbAll[dwRI * dwProbCol + dwCI];
            dwMaxIdx = 0;
            for (dwRI2 = 0; dwRI2 < dwProbRow; dwRI2++) {
                fTmp = pfProbCol1[dwRI2] + pfTrans[dwRI2 * dwProbRow + dwRI] + pfProbAll[dwRI * dwProbCol + dwCI];
                if (fMaxScore < fTmp) {
                    fMaxScore = fTmp;
                    dwMaxIdx = dwRI2;
                }
            }
            pfProbCol2[dwRI] = fMaxScore;
            pdwPathNet[dwRI * dwProbCol +  dwCI] = dwMaxIdx;
        }
        memcpy(pfProbCol1, pfProbCol2, dwProbRow * sizeof(float));
    }
    fMaxScore = pfProbCol2[0];
    dwMaxIdx = 0;
    for (dwRI = 0; dwRI < dwProbRow; dwRI++) {
        if (fMaxScore < pfProbCol2[dwRI]) {
            fMaxScore = pfProbCol2[dwRI];
            dwMaxIdx = dwRI;
        }
    }
    
    *pfScore = fMaxScore;
    pdwStateChainOut[dwProbCol - 1] = dwMaxIdx;
    for (dwCI = dwProbCol - 1; dwCI > 0; dwCI--) {
        pdwStateChainOut[dwCI - 1] = pdwPathNet[pdwStateChainOut[dwCI] * dwProbCol + dwCI];
    }
    
    free(pdwPathNet);
    free(pfProbCol1);
    free(pfProbCol2);
    
    return 0;
}



int IMP_LogNHmmGmm_Create(int dwNeedStateNum, IMP_LOGNHMMGMM_S *pstLogNHmmGmm)
{
    int dwI;
    
    pstLogNHmmGmm->dwStateNum = dwNeedStateNum;
    pstLogNHmmGmm->pfPrior = (float*)malloc(dwNeedStateNum * sizeof(float));
    pstLogNHmmGmm->pfTransfer = (float*)malloc(dwNeedStateNum * dwNeedStateNum * sizeof(float));
    
 //   memset(pstLogNHmmGmm->pfPrior, 0, dwNeedStateNum * sizeof(float));
 //   pstLogNHmmGmm->pfPrior[0] = 1.0f;
    for (dwI = 0; dwI < dwNeedStateNum; dwI++) {
        pstLogNHmmGmm->pfPrior[dwI] = LOGMINIMUM;
    }
    pstLogNHmmGmm->pfPrior[0] = 0;
    
    memset(pstLogNHmmGmm->pfTransfer, 0, dwNeedStateNum * dwNeedStateNum * sizeof(float));
    
    for (dwI = 0; dwI < dwNeedStateNum * dwNeedStateNum; dwI++) {
        pstLogNHmmGmm->pfTransfer[dwI] = LOGMINIMUM;
    }
    for (dwI = 0; dwI < dwNeedStateNum-1; dwI++) {
        pstLogNHmmGmm->pfTransfer[dwI * dwNeedStateNum + dwI + 1] = 0;
    }
    pstLogNHmmGmm->pfTransfer[0] = 0.69314f; //log(0.5)
    pstLogNHmmGmm->pfTransfer[1] = 0.69314f;
    pstLogNHmmGmm->pfTransfer[(dwNeedStateNum - 1) * dwNeedStateNum + 0] = 0.69314f;
    pstLogNHmmGmm->pfTransfer[(dwNeedStateNum - 1) * dwNeedStateNum + 1] = 0.69314f;
    
    return 0;
}



int IMP_LogNHmmGmm_Process(float *pfProbAll, int adwProbAllSize[2], IMP_LOGNHMMGMM_S *pstLogNHmmGmm, int *pdwStateChainOut, int dwChainLen, float *pfScore)
{
    imp_doViterbi(pfProbAll, pstLogNHmmGmm->dwStateNum, adwProbAllSize[1], pstLogNHmmGmm->pfPrior, pstLogNHmmGmm->pfTransfer, pdwStateChainOut, dwChainLen, pfScore);
    
    return 0;
}



int IMP_LogNHmmGmm_Exit(IMP_LOGNHMMGMM_S *pstLogNHmmGmm)
{
    free(pstLogNHmmGmm->pfPrior);
    free(pstLogNHmmGmm->pfTransfer);
    
    return 0;
}


