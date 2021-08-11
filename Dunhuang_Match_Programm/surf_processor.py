# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 13:05:13 2020

@author: 龚明泽(Rouch MacTavish)from CN SZU Digital Creative Technology Research Center
"""

import numpy as np 
import cv2
from matplotlib import pyplot as plt
import time 

def sift_processor (img1 , img2 ) :
    date = time.strftime('%Y-%m-%d',time.localtime(time.time())) ;
    start =  time.perf_counter() ;
    MIN_MATCH_COUNT = 10 ;

    surf = cv2.xfeatures2d.SURF_create(float(4000)) ;
    kp1 , des1 = sift.detectAndCompute (img1 , None ) ;
    kp2 , des2 = sift.detectAndCompute (img2 , None ) ;
    
    FLANN_INDEX_KDTREE = 0 ;
    index_para = dict(algorithm = FLANN_INDEX_KDTREE , trees = 5) ;
    search_para = dict (checks = 50) ;
    
    flann = cv2.FlannBasedMatcher (index_para , search_para) ;
    
    matches = flann.knnMatch (des1, des2 , k = 2 ) ;
    
    success = [] ;
    for m , n in matches :
        if m.distance < 0.7 * n.distance :
            success.append (m) ;
    
    if len(success ) > MIN_MATCH_COUNT :
        src_pts = np.float32 ([ kp1[m.queryIdx].pt for m in success]).reshape(-1,1,2) ;
        dst_pts = np.float32 ([ kp2[m.trainIdx].pt for m in success]).reshape(-1,1,2) ;
    
        M , mask = cv2.findHomography (src_pts , dst_pts , cv2.RANSAC , 5.0) ;
        matchesMask = mask.ravel().tolist() ;
    
        h , w = img1.shape ;
        pts = np.float32 ([[0 , 0] , [0 , h-1] , [w-1 , h-1] , [w-1 , 0]]).reshape(-1,1,2) ;
        dst = cv2.perspectiveTransform (pts , M ) ;
    
        img2 = cv2.polylines (img2 , [np.int32(dst)] , True , 255 , 3 , cv2.LINE_AA) ;
    
    else :
        print ("Not enough matches are found - %d / %d " % (len(success) , MIN_MATCH_COUNT) ) ;
        matchMask = None ;
        
    cv2.imwrite ('E:\Programm of Img\Result\SURF\Detected' + str(date) +'.jpg' ,  img2 ) ;
    end = time.perf_counter() ;
    time_out = end - start ;
    print ("timeout = " + str(time_out)) ;
    print ("accuracy : " + str (len(success) / len(kp1))) ;
    return time_out ,  ( len(success) / len(kp1) );
