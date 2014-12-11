//
//  lnhgLPCE.h
//  simpleCNN
//
//  Created by mzhang on 14/12/11.
//  Copyright (c) 2014年 ___STAR___. All rights reserved.
//

#ifndef __simpleCNN__lnhgLPCE__
#define __simpleCNN__lnhgLPCE__

#include <stdio.h>
#include "stdlib.h"
#include "string.h"

#define WIN_W 14
#define WIN_H 32

typedef unsigned char uchar;


int IMP_LNHG_LPCE_Create();

int IMP_LNHG_LPCE_Process(uchar *pubyImg, int dwImgW, int dwImgH);

int IMP_LNHG_LPCE_Exit();


#endif /* defined(__simpleCNN__lnhgLPCE__) */
