import numpy as np
import random
import cv2
from matplotlib import pyplot as plt
np.set_printoptions(threshold=np.nan)
#used variables
lowerskinHSV = np.array([0,48,80],np.uint8)
upperskinHSV = np.array([20,255,255],np.uint8)

lowerskinYCC = np.array([80,135,85],np.uint8)
upperskinYCC = np.array([255,180,135],np.uint8)





def backProject(mask,frame,bins,hist):
    f=frame.shape
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    if np.size(mask)>1:
        hist = cv2.calcHist([frame_hsv],[0,1],mask,[bins]*2,[0,181,0,256])
    hist2 = cv2.calcHist([frame_hsv],[0,1],None,[bins]*2,[0,181,0,256]) 
    hist_norm=hist*255/np.sum(hist)
    hist2_norm=hist2*255/np.sum(hist2)
    p_skin = cv2.calcBackProject([frame_hsv],[0,1],hist_norm,[0,181,0,256],1)
    p_all = cv2.calcBackProject([frame_hsv],[0,1],hist2_norm,[0,181,0,256],1)
    p_all[p_all==0]=1
    p_all=p_all.astype(float)/255
    p_skin=p_skin.astype(float)/255
    dst=p_skin/p_all
    dst=dst*255/np.max(dst)
    dst=np.uint8(dst)
    dst=np.uint8(255*(dst>50))
    dst=cv2.medianBlur(dst,9)
    hist = cv2.calcHist([frame_hsv],[0,1],dst,[bins]*2,[0,181,0,256])
    return dst,hist



def movement(frame1,frame2,thresh):
    D=cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)-cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    mask=(np.absolute(D)>np.mean(frame1)*thresh)*1.0
    mask = cv2.erode(mask,None,iterations = 3)
    mask = cv2.dilate(mask,None,iterations = 5)
    mask=np.uint8(mask*255)
    mask=cv2.medianBlur(mask,15)
    return mask

def findHands(frame1,frame2,hist):
    skin,move=movingSkin(frame1,frame2)
    mask=skin&move
    score=np.mean(mask)/255
    if score>0.02:
        print("motion")
        ms,newhist=backProject(mask,frame1,24,0)
        res = cv2.dilate(ms,None,iterations = 2)
        if np.size(hist)>1:
            d=cv2.compareHist( hist,newhist, cv2.cv.CV_COMP_CORREL)
            if d>0.75:
                return res,newhist
            print(d)
            print("blocked")
            show(res)
        else:
            return res,newhist
    if np.size(hist)>1:
        print("guess")
        ms,newhist=backProject(0,frame1,24,hist)
        res = cv2.dilate(ms,None,iterations = 2)
        return res,newhist       
    return 0,0


    
def movingSkin(frame1,frame2):
    move=movement(frame1,frame2,0.5)
    skin=skinColor(frame1)
    return skin,move

def skinColor(frame):
    bins=256
    HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    Ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
    mask1 = cv2.inRange(HSV, lowerskinHSV, upperskinHSV)
    mask2 = cv2.inRange(Ycrcb, lowerskinYCC, upperskinYCC)
    show(mask1)
    print(" ---------------------")
    return mask1 & mask2

def show(img):
    cv2.imshow('result',img)
    cv2.waitKey(0)
    return

#reuslt= resultmask, histogram
def handMask(frame1,frame2):
    frame1+cv2.resize(frame2, (200,200))
    frame2=cv2.resize(frame2, (200,200))
    hist=0
    return findHands(frame,previous,hist)
    
    






