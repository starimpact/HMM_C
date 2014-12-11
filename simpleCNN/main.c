//
//  main.c
//  simpleCNN
//
//  Created by mzhang on 14/12/8.
//  Copyright (c) 2014年 ___STAR___. All rights reserved.
//

#include <stdio.h>
#include "cv.h"
#include "highgui.h"

#include "lnhgLPCE.h"


int main(int argc, const char * argv[]) {
    int dwI;
    IplImage *pstImage = 0;
//    char *pbyFN = "/Users/mzhang/work/HMM/image/";
    char *pbyFN = "/Users/mzhang/Downloads/plate/img2/";
    char abyFN[512];

    for (dwI = 0; dwI < 100; dwI++) {
        sprintf(abyFN, "%s%d.bmp", pbyFN, dwI);
        pstImage = cvLoadImage(abyFN, CV_LOAD_IMAGE_GRAYSCALE);
        if (!pstImage) {
            printf("can't read image.\n");
            continue;
        }
        else
        {
            int dwRI;
            int dwW = pstImage->width, dwH = pstImage->height;
            uchar *pubyImg = (uchar*)malloc(dwW * dwH);
            
            cvShowImage("origin", pstImage);
            
            
            for (dwRI = 0; dwRI < dwH; dwRI++) {
                memcpy(pubyImg + dwRI * dwW, pstImage->imageData + dwRI * pstImage->widthStep, dwW);
            }
            
            IMP_LNHG_LPCE_Process(pubyImg, dwW, dwH);
            
            free(pubyImg);
            cvReleaseImage(&pstImage);
            
            cvWaitKey(0);
        }
    }
    
    
    
    return 0;
}



