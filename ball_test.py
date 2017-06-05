import numpy as np
import cv2
from matplotlib import pyplot as plt
import cups_colored as cc
import handfeatures as hf
import os


lowerball = np.array([0,175,170],np.uint8)
upperball = np.array([10,255,255],np.uint8)
lowerball2 = np.array([170,175,170],np.uint8)
upperball2 = np.array([181,255,255],np.uint8)
calibrateArea=[0,0]


def show(frame):
    cv2.imshow('show',frame)
    cv2.waitKey(0)


def ball_mask(frame):
    # convert to hsv colorspace
    converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #remove small errors in background
    mask1 = cv2.inRange(converted, lowerball, upperball)
    mask2 = cv2.inRange(converted, lowerball2, upperball2)
    mask=mask1|mask2
    mask=cv2.erode(mask,None,5)
    mask=cv2.dilate(mask,None,7)
    mask = cv2.medianBlur(mask,3)
   # show(mask)
    return mask

def maxInscribedCircle(cnt):
    maxdist=-1
    (x, y, w, h) = cv2.boundingRect(cnt)
    for i in range(x,x+w):
        for j in range(y,y+h):
            dist=cv2.pointPolygonTest(cnt, (i,j), True)
            if (dist>maxdist):
                maxdist=dist
                center=(i,j)
    return center,maxdist


def testball(frame):
    num=0
    mask=ball_mask(frame)
    copymask=np.copy(mask)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    (cnts, cmask) = cv2.findContours(mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    locs=[]
    i=0;
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        area=cv2.contourArea(c)/(frame.shape[0]*frame.shape[1])
        if area>0.0005:
            i+=1
            center,radius=maxInscribedCircle(c)
            score=cv2.contourArea(c)/(3.14*radius**2)
          #  print(score)
            t=[2.5,1.87]
            if( score> t[0] ):
                num+=3
            elif( score>t[1]):       
                num+=2
            else:
                calibrateArea[0]+=cv2.contourArea(c)
                calibrateArea[1]+=1
                num+=1
    #show(frame)
    show(copymask)
    return num%4,score

def testball2(frame):
    num=0
    mask=ball_mask(frame)
    copymask=np.copy(mask)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    (cnts, cmask) = cv2.findContours(mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    locs=[]
    i=0;
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        area=cv2.contourArea(c)/(frame.shape[0]*frame.shape[1])
        if area>0.0005:
            i+=1
            center,radius=maxInscribedCircle(c)
            score=calibrateArea[1]*cv2.contourArea(c)/calibrateArea[0]
           # print(score)
            t=[2.22,1.26]
            if( score> t[0] ):
                num+=3
            elif( score>t[1]):
                num+=2
            else:
                num+=1
    return num%4,score



def resize(frame,width):
    h,w,d=frame.shape
    frame=cv2.resize(frame,(width,int(width*h/w)))
    return frame

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            img=resize(img,300)
            cv2.imwrite(os.path.join(folder,filename),img)
            images.append(img)
    return images


    
folders=["one","two","three"]  
results=np.zeros((3,3))
mean=0
for f in [0,1,2]:
    scores=[]
    for i in load_images_from_folder("balls/"+folders[f]):
        r=testball(i)
        results[f,r[0]-1]+=1
        scores.append(r[1])
    print(min(scores),max(scores),np.mean(np.array(scores)))
    print(mean+(np.mean(np.array(scores))-mean)/2)
    mean=np.mean(np.array(scores))
print(results)
print((results[0,0]+results[1,1]+results[2,2])/np.sum(results))
mean=0
results=np.zeros((3,3))   
for f in [0,1,2]:
    scores=[]
    for i in load_images_from_folder("balls/"+folders[f]):
        r=testball2(i)
        results[f,r[0]-1]+=1
        scores.append(r[1])
    print(min(scores),max(scores),np.mean(np.array(scores)))
    print(mean+(np.mean(np.array(scores))-mean)/2)
    mean=np.mean(np.array(scores))
print(results)
print(np.sum(results))
print((results[0,0]+results[1,1]+results[2,2])/np.sum(results))
