# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 15:16:34 2020

@author: Rouch MacTavish
Img Data Generating Programm On Windows
"""

import os
import random
import math
from multiprocessing import cpu_count
import cv2
import image_augmentation as ia

class Augmentation_Parameters() :
    def __init__ (self , num , num_procs , p_mirror , p_crop , crop_size , crop_hw_vari , p_rotate , p_rotate_crop,
                  rotate_angle_vari , p_hsv , hue_vari , sat_vari , val_vari , p_gamma , gamma_vari) :
        self.num = num ;
        self.num_procs = num_procs ;
        self.p_mirror= p_mirror ;
        self.p_crop = p_crop ;
        self.crop_size = crop_size ;
        self.crop_hw_vari = crop_hw_vari ;
        self.p_rotate = p_rotate ;
        self.p_rotate_crop = p_rotate_crop ;
        self.rotate_angle_vari = rotate_angle_vari ;
        self.p_hsv = p_hsv ;
        self.hue_vari = hue_vari ;
        self.sat_vari = sat_vari ;
        self.val_vari = val_vari ;
        self.p_gamma = p_gamma ;
        self.gamma_vari = gamma_vari ;
        
    def update_para (self , snum , new_para) :
        if snum == 1 :
            self.num = new_para;
        elif snum == 2 :
            self.num_procs = new_para ;
        elif snum == 3 :
            self.p_mirro = new_para ;
        elif snum == 5 :
            self.crop_size = new_para ;
        elif snum == 4 :
            self.p_crop = new_para ;
        elif snum == 6 :
            self.crop_hw_vari = new_para ;
        elif snum == 7 :
            self.p_rotate = new_para ;
        elif snum == 8 :
            self.rotate_angle_vari = new_para ;
        elif snum == 9 :
            self.p_hsv = new_para ;
        elif snum == 10 :
            self.hue_vari = new_para ;
        elif snum == 11 :
            self.sat_vari = new_para ;
        elif snum == 12 :
            self.val_vari = new_para ;
        elif snum == 13 :
            self.p_gamma = new_para ;
        elif snum == 14 :
            self.gamma_vari = new_para ;
            
def directions () :
    lib1 = ['num' , 'num_procs' , 'p_mirror' , 'p_crop' , 'crop_size' , 'crop_hw_vari' , 'p_rotate' , 'p_rotate_crop' 
           'rotate_angle_vari' , 'p_hsv' , 'hue_vari' , 'sat_vari' , 'val_vari' , 'p_gamma' , 
           'gamma_vari'] ;
    lib2 = [208*30, int(cpu_count()) , float(0.5) , float(1.0) , float (0.8) , float (0.1) , float (1.0) , 
            float (0.5) , float (180.0) , float (1.0) , int (10) , float (0.1) , float (0.1) , float (1.0) ,
            float (2.0)] ;
    print ("This is a simple tool for image data augmentation ! ") ;
    print ("Here are the original parameters and their serial number !") ;
    for i in range (len(lib1)) :
        print (i+1 , ":" , lib1[i] , "=" , lib2[i]) ;
    
    print ("if you want to change these parameters , you can use a function named update_para(serial_number , new_para) , there will be an asking for checking out if you want to change the parameters and you just need to input the serial number and new parameter") ;

    return lib1 , lib2 ;

def cmd_input (para) :
    flag = 1 ;
    while (flag) :
        print ("Do you want to update the parameters ?") ;
        mark = input ("If you want to do so , please input Y ! Otherwise input N !") ;
        if mark == 'Y' or mark == 'y' :
            index = input ("Please input the index of the parameter : ");
            value = input ("Please input the new value of the parameter :");
            para.update_para (index , value ) ;
            flag = 1 ;
            print ("Update Successfully ! ");
        else :
            flag = 0 ;

    return para ;

def generate_image_list(para , input_dir , output_dir):
    filenames = os.listdir(input_dir) ;
    num_imgs = len(filenames) ;

    num_ave_aug = int(math.floor(para.num/num_imgs)) ;
    rem = para.num - num_ave_aug*num_imgs ;
    lucky_seq = [True]*rem + [False]*(num_imgs-rem) ;
    random.shuffle(lucky_seq) ;

    img_list = [
        (os.sep.join([input_dir, filename]), num_ave_aug+1 if lucky else num_ave_aug)
        for filename, lucky in zip(filenames, lucky_seq)
    ] ;

    random.shuffle(img_list)  ; 

    length = float(num_imgs) / float(para.num_procs) ;
    indices = [int(round(i * length)) for i in range(para.num_procs + 1)] ;
    return [img_list[indices[i]:indices[i + 1]] for i in range(para.num_procs)] ;

def augment_images(filelist, para , output_dir):
    for filepath, n in filelist :
        img = cv2.imread(filepath) ;
        filename = filepath.split(os.sep)[-1] ;
        dot_pos = filename.rfind('.') ;
        imgname = filename[:dot_pos] ;
        ext = filename[dot_pos:] ;

        print('Augmenting {} ...'.format(filename )) ;
        for i in range(n) :
            img_varied = img.copy() ;
            varied_imgname = '{}_{:0>3d}_'.format(imgname, i) ;
            if random.random() < para.p_mirror :
                img_varied = cv2.flip(img_varied, 1) ;
                varied_imgname += 'm' ;
                
            if random.random() < para.p_crop :
                img_varied = ia.random_crop(
                    img_varied,
                    para.crop_size,
                    para.crop_hw_vari) ;
                varied_imgname += 'c' ;
                
            if random.random() < para.p_rotate :
                img_varied = ia.random_rotate(
                    img_varied,
                    para.rotate_angle_vari,
                    para.p_rotate_crop) ;
                varied_imgname += 'r' ;
                
            if random.random() < para.p_hsv :
                img_varied = ia.random_hsv_transform(
                    img_varied , 
                    para.hue_vari , 
                    para.sat_vari , 
                    para.val_vari) ;
                varied_imgname += 'h' ;
                
            if random.random() < para.p_gamma :
                img_varied = ia.random_gamma_transform(
                    img_varied , 
                    para.gamma_vari) ;
                varied_imgname += 'g' ;
                
            output_filepath = os.sep.join([
                output_dir , 
                '{}{}'.format(varied_imgname , ext)]) ;
            cv2.imwrite (output_filepath , img_varied) ;
            
def main () :
    #Normoalize Parameters 
    index , value = directions() ;
    para = Augmentation_Parameters(0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ,0 , 0 , 0 , 0 , 0) ;
    #Get diractories  
    input_dir = input ("Please input the diractory of the images waiting to be augmentated : ") ;
    output_dir = input ("Please input the target diractory to store the augmentated images : ") ;
    if not os.path.exists(output_dir ) :
        os.mkdir(output_dir ) ;
    print ("Ready to Go ") ;
    #Put values into class
    for i in range ( len(index) ) :
        para.update_para ( i+1 , value[i] ) ;
    #Check out and update
    para = cmd_input(para) ;
    #Generate image list
    sublists = generate_image_list (para, input_dir , output_dir ) ;
    #Processed
    flag = 1;
    for j in sublists :
        print ("Epoch : " + str (flag));
        augment_images ( j , para , output_dir) ;
    return ;

main() ;
