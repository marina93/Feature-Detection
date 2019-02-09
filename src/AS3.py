#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 11:47:02 2018

@author: marina
"""

import cv2
import numpy as np
import math
import sys

from numpy.linalg import inv

args = sys.argv
imgOrg =  np.zeros((5,5), np.uint8)

def run():
    print("Please insert a command. Press 'help' to see the options")
    value = raw_input("prompt: ")
    value_list = value.split()
    if((len(value_list)==1) and value_list[0]=='corner1'):
        cornerDetector1(getImageFromCamera())
    if(len(value_list)==3and value_list[0]=='corner1'):
        cornerDetector1(value_list[1])
    if((len(value_list)==1) and value_list[0]=='features'):
        features(getImageFromCamera(),getImageFromCamera())
    if(len(value_list)==3and value_list[0]=='features'):
        features(value_list[1], value_list[2])
    
    while(1):
         if value == 'help':
             h()
         if value == 'q':
             break
         
def getImage(imageName):
    
    global imgOrg
    imgOrg = cv2.imread(imageName)
    return imgOrg

def getImageFromCamera():
    global cam, imgOrg
    cam = 1
    cam = cv2.VideoCapture(0)
    
    while(1):
        ret,frame = cam.read()
        if not ret:
            break
        imgOrg = frame
    return imgOrg

def grayScale(image):
    global imgOrg
    imgOrg = image.copy()
    imgGray = cv2.cvtColor(imgOrg, cv2.COLOR_RGB2GRAY)
    return imgGray

    
def scaleImage(image):
    global imgOrg

    imgOrg = image.copy()
    img = cv2.pyrDown(imgOrg)
    
    if (img.shape[0]>500):
        img = cv2.pyrDown(img)
    return img 

def nothing():
    pass

def cornerDetector1(image1):
    
    global imgMod, imgOrg
    
    img1 = getImage(image1)
    
    if((img1.shape[0]>500)or (img1.shape[1]>500)):
        img1 = scaleImage(img1)
    img1 = img1.copy()
    
    if((img1.shape[0]>500)or (img1.shape[1]>500)):
        img1 = scaleImage(img1)
    img1 = img1.copy()
    
    img1 = grayScale(img1)
    
    
    cv2.namedWindow('Corner Detection 1',0)

    cv2.createTrackbar('ksize','Corner Detection 1',3,7,nothing)

    cv2.createTrackbar('Window Size','Corner Detection 1',2,6,nothing)

    cv2.createTrackbar('k','Corner Detection 1',1,3,nothing)

    cv2.createTrackbar('Threshold','Corner Detection 1',10,1000,nothing)

    while(True):
        
        ksize1 = cv2.getTrackbarPos('ksize','Corner Detection 1')
        
        windowSize1 = cv2.getTrackbarPos('Window Size','Corner Detection 1')
        
        k1 = cv2.getTrackbarPos('k','Corner Detection 1')
        
        threshold1 = cv2.getTrackbarPos('Threshold','Corner Detection 1')
        
        cv2.imshow('Corner Detection 1',harrisAlgth(ksize1, img1, windowSize1, k1, threshold1))

        value = cv2.waitKey(50)&0xff
        if(value == ord('q')):
            cv2.destroyAllWindows()
            #run()
            break
        
def harrisAlgth(ksize, image, windowSize, k, threshold):
    global imgMod, imgOrg
    
    img = image.copy()
    if((ksize%2)==0):
        ksize=ksize+1
    if k==1:
        k=0.04
    elif k==2:
        k=0.1
    elif k==3:
        k=0.2
    
    threshold = threshold*100000000
    dx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize)
    dy = cv2.Sobel(img,cv2.CV_64F,0,1,ksize)
   
    Ixx = dx*dx
    Ixy = dx*dy
    Iyy = dy*dy
    
    corners = []
    
    h = img.shape[0]
    w = img.shape[1]
    offs =windowSize/2
    
    for y in range(offs, h-offs ):
        for x in range(offs, w-offs):
            
            compIxx = Ixx[y-offs:y+offs+1, x-offs:x+offs+1]
            compIxy = Ixy[y-offs:y+offs+1, x-offs:x+offs+1]
            compIyy = Iyy[y-offs:y+offs+1, x-offs:x+offs+1]

            Corrxx = compIxx.sum()
            Corrxy = compIxy.sum()
         #   Corryx = Corrxy
            Corryy = compIyy.sum()
            
         

            det = (Corrxx*Corryy) - (Corrxy*Corrxy)
            tr = Corrxx + Corryy
            C_c = det - k*tr*tr
            
            if C_c > (threshold):
                print("Threshold")
                corners.append([y,x,C_c])
                
                cv2.rectangle(img,(x-10,y-10),(x+10,y+10),(0,0,255),2,5)
       
    print("Out")           
    return img


def features(image1, image2):
    COUNT = 8
    
    global imgMod, imgOrg
    
    image1 = getImage(image1)
    image2 = getImage(image2)

    image1 = image1.copy()
    image2 = image2.copy()
    
    if((image1.shape[0]>500)or (image1.shape[1]>500)):
        image1 = scaleImage(image1)
    image1 = image1.copy()
    
    if((image2.shape[0]>500)or (image2.shape[1]>500)):
        image2 = scaleImage(image2)
    image2 = image2.copy()
    
    if((image1.shape[0]>500)or (image1.shape[1]>500)):
        image1 = scaleImage(image1)
    image1 = image1.copy()
    
    if((image2.shape[0]>500)or (image2.shape[1]>500)):
        image2 = scaleImage(image2)
    image2 = image2.copy()

    sift = cv2.xfeatures2d.SIFT_create()
    FLANN_INDEX_KDTREE = 0
    index = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search = dict(checks = 50)
    
    kp1, des1 = sift.detectAndCompute(image1, None)
    kp2, des2 = sift.detectAndCompute(image2, None)
    
    
    
    f = cv2.FlannBasedMatcher(index, search)
    matchPoints = f.knnMatch(des1,des2,k=2)
    
    vals = []
    for m,n in matchPoints:
        if m.distance <0.7*n.distance:
            vals.append(m)

    
    if len(vals)>COUNT:
        source = np.float32([ kp1[m.queryIdx].pt for m in vals ]).reshape(-1,1,2)
        dest = np.float32([ kp2[m.trainIdx].pt for m in vals ]).reshape(-1,1,2)
        
        M, mask = cv2.findHomography(source, dest, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()

        h = image1.shape[0]
        w = image1.shape[1]
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)
    
        image2 = cv2.polylines(image2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
    
    else:
        nothing
    
    draw_params = dict(matchColor = (0,0,255),
                        singlePointColor = None,
                        matchesMask = matchesMask,
                        flags = 2)
     
    img3 = cv2.drawMatches(image1,kp1,image2,kp2,vals,None,**draw_params)
    img3 = grayScale(img3)
    
    cv2.imshow('Features match', img3)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 
    return img3

def h():
    print("Program description: ")
    print("q: quit a function")
    print("corner1: To detect corners using the implementation of Harris algorithm (without openCV). If no images are specified, the images will be capture from the camera.")
    print("features:  Matches feature vectors between images.If no images are specified, the images will be capture from the camera. " )
    run()