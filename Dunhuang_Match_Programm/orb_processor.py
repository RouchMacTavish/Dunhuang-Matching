# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 13:05:13 2020

@author: 龚明泽(Rouch MacTavish)from CN SZU Digital Creative Technology Research Center
"""

import numpy as np 
import cv2
from matplotlib import pyplot as plt
import time

def orb_processor (img1 , img2 ) :
    date = time.strftime('%Y-%m-%d',time.localtime(time.time())) ;
    start = time.perf_counter() ;
    MIN_MATCH_COUNT = 10 ;
    
    orb = cv2.ORB_create()
    kp3 , des3 = orb.detectAndCompute(img1, None ) ;
    kp4 , des4 = orb.detectAndCompute(img2, None ) ;
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING , crossCheck = True) ;
    matches = bf.match(des3,des4) ;
    matches = sorted (matches , key = lambda x:x.distance) ;
    
    if len(matches) > MIN_MATCH_COUNT :
        src_pts = np.float32 ([ kp3[m.queryIdx].pt for m in matches]).reshape(-1,1,2) ;
        dst_pts = np.float32 ([ kp4[m.trainIdx].pt for m in matches]).reshape(-1,1,2) ;
    
        M , mask = cv2.findHomography (src_pts , dst_pts , cv2.RANSAC , 5.0) ;
    
        h , w = img1.shape ;
        pts = np.float32 ([[0 , 0] , [0 , h-1] , [w-1 , h-1] , [w-1 , 0]]).reshape(-1,1,2) ;
        dst = cv2.perspectiveTransform (pts , M ) ;
    
        img2 = cv2.polylines (img2 , [np.int32(dst)] , True , 255 , 3 , cv2.LINE_AA) ;
    
    else :
        print ("Not enough matches are found - %d / %d " % (len(matches) , MIN_MATCH_COUNT) ) ;

    cv2.imwrite ('E:\Programm of Img\Result\ORB\Detected' + str(date) +'.jpg' ,  img2 ) ;
    end = time.perf_counter () ;
    time_out = end - start ;
    print ("timeout = " + str(time_out)) ;
    print ("accuracy : " + str (len(success) / len(kp1))) ;
    
    return time_out ,  ( len(success) / len(kp1) );