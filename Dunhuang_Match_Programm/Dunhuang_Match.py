# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 14:49:06 2020

@author: 龚明泽(Rouch MacTavish)from CN SZU Digital Creative Technology Research Center
"""

import cv2 
import numpy 
import os
from os import walk 
from os.path import join 
import csv 
from datetime import datetime

def get_features (document):
    files2 = [] ;
    for (root , dirs , files ) in walk (document):
        files2.extend (files) ; #Walk through the folder 
    for file in files :
        if '.jpg' in file or '.JPG' in file :
            #If you want to use SURF,just cancel the "#" in the following cade segment,the same as SURF
            #save _featuresimf (document , file ,cv2.xfeatures2d.SIFT_creat() ) ;
            save_featuresimf (document , file , cv2.xfeatures2d.SURF_create(float (4000)) ) ; #Save the imformation of features 
        
def save_featuresimf (document , path , detector ):#This detector should be SIFT or SURF 
    if path.endswith("npy"):
        return; #Just  to make sure that's a image instead of npy
    img = cv2.imread (join(document , path) , cv2.IMREAD_ANYCOLOR) ;
    keypoint , descriptor = detector.detectAndCompute (img , None ) ;
    file = path.replace ("jpg" , "npy") ; 
    numpy.save (join (document , file ) , descriptor ) ;
    #Get the descriptor of the image and save it in the file ends with "npy"

def get_npyfiles (document):
    descriptorgroup = [] ; 
    #Walk through the folder and gather all the files end with "npy" then return
    for (root , dirs , files ) in walk (document):
        for file in files :
            if file.endswith("npy"):
                descriptorgroup.append (file) ;
    return (descriptorgroup) ; 

def input_process (root , file ):
    imgname = root + "/" + file ;
    img = cv2.imread (imgname , cv2.IMREAD_ANYCOLOR);
    #sift = cv2.xfeatures2d.SIFT_create() ;
    gray = cv2.cvtColor (img,cv2.COLOR_BGR2GRAY);
    surf = cv2.xfeatures2d.SURF_create(float(4000)) ;
    keypoint , descriptor = surf.detectAndCompute (gray , None ) ;
    return (keypoint , descriptor) ;


def FLANN_Matcher (lib_dir , descriptorgroup , deascriptor , imgname ):
    #Build up 
    indexparams = dict (algorithm = 0 , trees = 5 );
    searchparams = dict (checks = 50 ) ;
    flann = cv2.FlannBasedMatcher(indexparams , searchparams ) ;
    #Matching
    candidates = {} ;
    for each_desc in descriptorgroup :
        matches  = flann.knnMatch(deascriptor , numpy.load(join (lib_dir , each_desc)) , k=2 ) ;
        success = [] ;
        for i , j in matches :
            if i.distance < 0.7 * j.distance :
                success.append (i) ;
        accurancy = len(success) / len(matches) ;
        accurancy = round (accurancy*100) ;
        #print ("%s  :  The accurancy is %d %%" % (each_desc , accurancy)) ;
        candidates [each_desc] = accurancy ;
        csv_write (imgname , each_desc , accurancy) ;
    
    #Find out the best_match
    best_matchnum = None ;
    best_match = None ;
    count = 0 ;
    flag = 0 ;
    for candidate , match in candidates.items() :
        flag = flag + 1 ;
        if match > 60 :
            count = count + 1 ;
        if best_matchnum == None or match > best_matchnum :
            best_matchnum = match ;
            best_match = candidate ;
    print ("The best match one is %s , in %s" % (best_match , lib_dir)) ;
    print ("\n") ;
    lib_match_rate= count / flag ;
    best_match_name = best_match.split('.')[0] ;
    best_match_name = best_match_name + ".jpg" ;
    return (best_matchnum,lib_match_rate ,best_match_name) ;

def csv_write ( imgname , MatchedImg , accurancy ) :
    csv_file = open ("Details.csv" , 'a') ;
    writer = csv.writer (csv_file) ;
    row = [] ;
    row.append (imgname) ;
    row.append (MatchedImg) ;
    row.append (accurancy) ;
    writer.writerow(row) ;
    csv_file.close () ;
    return 0 ;

def dunhuang_match (n , lib_dir , pic ) :
    best_match_rate = 0 ;
    inputimgpath = pic [0] ;
    inputimgname = pic [1] ;
    get_features (lib_dir) ;
    group = get_npyfiles (lib_dir) ;
    keypoint , descriptor = input_process( inputimgpath , inputimgname ) ;
    best_match_rate , lib_match_rate , best_match= FLANN_Matcher (lib_dir  ,group , descriptor , inputimgname ) ;
    print ("In Cave %s , This picture has the highest matching rate of %d %% with image %s" % (n , best_match_rate , best_match)) ;
    clock = datetime.now() ;
    txt = open("Result.txt" , "w+") ;
    txt.write(str(clock) + "In Cave %s , This picture has the highest matching rate of %d %% with image %s" % (n , best_match_rate , best_match)) ;
    txt.close() ;
    print ("Also , this picture has an accurancy of %d%% about the whole lib match rate" % (lib_match_rate*100)) ;
    
def main () :
    pic = ["inputimgpath" , "inputimgname"] ;
    #folder = ("E:\Programm of Img\Test Img\Trainning Data") ;
    folder = str(input ("Plz input the path of the databese : ")) ;
    flag = True ;
    pic [0] = str(input ("Plz input the root of the picture : ")) ;
    pic [1] = str(input ("Plz input the whole name of the picture :")) ;
    for root , filenames , files , in os.walk (folder) :
        break ;
    print (filenames) ;
    while flag :
        n = input ("Plz input the number of the cave you want to match : ") ;
        if n not in filenames :
            print ("Sry! There is no such a cave in database ! Plz try another one or just exit :P ") ;
        else :
            flag = False ;
        
    print ("\n") ;
    lib_dir = folder + "/" + str(n) ;
    dunhuang_match (n , lib_dir , pic ) ;
    return 0 ;

main ();
