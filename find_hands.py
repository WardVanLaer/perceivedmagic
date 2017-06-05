import numpy as np
import cv2
from matplotlib import pyplot as plt

np.set_printoptions(threshold=np.nan)
#used variables
lowerskinHSV1 = np.array([0,30,80],np.uint8)
upperskinHSV1 = np.array([15,255,255],np.uint8)

lowerskinHSV2 = np.array([170,30,80],np.uint8)
upperskinHSV2 = np.array([180,255,255],np.uint8)

lowerskinBGR = np.array([20,40,95],np.uint8)
upperskinBGR = np.array([255,255,255],np.uint8)


lowerskinYCC = np.array([80,135,85],np.uint8)
upperskinYCC = np.array([255,180,135],np.uint8)

bins=48

def backProject(mask,frame,bins,hist):
    show(mask)
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
    dst=np.uint8(255*(dst>80))
    dst=cv2.medianBlur(dst,5)
    show(dst)
    hist = cv2.calcHist([frame_hsv],[0,1],dst,[bins]*2,[0,181,0,256])
    return dst,hist


def movement(frame1,frame2,thresh):
    D=cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)-cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    mask=(np.absolute(D)>np.mean(frame1)*thresh)*1.0
    mask = cv2.erode(mask,None,iterations = 3)
    mask = cv2.dilate(mask,None,iterations = 7)
    mask=np.uint8(mask*255)
    mask=cv2.medianBlur(mask,15)
    return mask




    
def movingSkin(frame1,frame2):
    move=movement(frame1,frame2,0.5)
    skin=skinColor(frame1)
    return skin,move

def skinColor(frame):
    bins=256
    HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    Ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
    mask1 = np.uint8(cv2.inRange(HSV, lowerskinHSV1, upperskinHSV1) | cv2.inRange(HSV, lowerskinHSV2, upperskinHSV2))
    mask2 = cv2.inRange(Ycrcb, lowerskinYCC, upperskinYCC)
    bgr1 = cv2.inRange(frame, lowerskinBGR, upperskinBGR)
    bgr2=((np.amax(frame,axis=2)-np.amin(frame,axis=2))>15)*255
    bgr3= ((frame[:,:,0]<frame[:,:,2]) & (frame[:,:,1]<frame[:,:,2]))*255
    bgr4= (np.abs(frame[:,:,1]-frame[:,:,2])>15)*255
    mask3=np.uint8(bgr3 & bgr2 & bgr1 & bgr4)
    result=cv2.erode(mask1 & mask3,None,2)
   # result=cv2.dilate(result,3)
    result=cv2.medianBlur(result,3)
    return result

def show(img):
    cv2.imshow('show',img)
    cv2.waitKey(0)
    return



class handSubstractor:
    def __init__(self,frame):
        frame=cv2.resize(frame, (200,200))
        self.previous=frame
        self.hist=0
        self.history=np.uint8(np.ones((200,200)))
        self.block_count=0
    def next(self,frame):
        frame=cv2.resize(frame, (200,200))
        res,self.hist=self.hand_mask(self.previous,frame,self.hist)
        self.previous=frame
        smooth= res & self.history 
        self.history=res
        cv2.imshow('result',smooth)
        return smooth      
       
    def hand_mask(self,frame1,frame2,hist):
        print(self.block_count)
        skin,move=movingSkin(frame1,frame2)
        mask=skin
        show(mask)
        score=np.mean(mask)/255
        if score>0.01:
            ms,newhist=backProject(mask,frame1,bins,0)
            res = cv2.dilate(mask,None,iterations = 3)
            res=cv2.medianBlur(res,9)
            if np.size(hist)>1:
                d=cv2.compareHist( hist,newhist, cv2.cv.CV_COMP_CORREL)
                if d>0.75 or self.block_count>5:
                    self.block_count=0
                    print("motion")
                    return res,newhist
                self.block_count+=1               
                print("hand recognition: renewal histogram blocked with score "+str(d))
            else:
                return res,newhist
        if np.size(hist)>1:
            print("guess")
            ms,newhist=backProject(0,frame1,bins,hist)
            res,newhist=backProject(ms,frame1,bins,hist)
         #   res = cv2.dilate(ms,None,iterations = 3)
          #  res=cv2.medianBlur(res,9)
            return res,newhist       
        return np.zeros_like(mask),0 
    






