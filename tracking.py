import numpy as np
import cv2
import imutils
from matplotlib import pyplot as plt
from skimage.measure import compare_ssim as ssim

cap= cv2.VideoCapture('vids/test1.mov')
HIST_SIZE=10
HIST_THRESH=100
ret,previous=cap.read()
previous=cv2.resize(previous, (200,130), fx=0.5, fy=0.5)
loc1=[(30,80),(55,105)]
loc2=[(80,75),(105,100)]
objects=[loc1,loc2]
histograms=[]


def rotateImage(image, angle,center):
  rot_mat = cv2.getRotationMatrix2D(center,angle,1.0)
  result = cv2.warpAffine(image, rot_mat, (image.shape[1],image.shape[0]),flags=cv2.INTER_LINEAR)
  return result


def rotatedRect(img,loc,theta):
    start,stop=loc
    size=(stop[0]-start[0],stop[1]-start[1])
    rect=((start[0]+size[0]/2,start[1]+size[1]/2),size,theta)
    box = cv2.cv.BoxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(img,[box],0,(0,0,255),2)
    return img

def show(img):
    cv2.imshow('imshow',img)
    cv2.waitKey(0)

def trackObject(location,previous,now):
    previous=cv2.cvtColor(previous,cv2.COLOR_BGR2GRAY)
    now=cv2.cvtColor(now,cv2.COLOR_BGR2GRAY)
    start=location[0]
    stop=location[1]
    size=stop[0]-start[0],stop[1]-start[1]
    center=start[0]+size[0]/2,start[1]+size[1]/2
    template=previous[start[1]:stop[1],start[0]:stop[0]]
    for angle in [10,0]:
        for step in [8,4,2,1]:
            now=rotateImage(now,5,center)
            start,stop=SS(step,template,now,start)
    return (start,stop)
    

def SS(step,template,image,location):
    best=[(0,0),(0,0),float("inf")]
    s1,s2=template.shape[0:2]
    options=[0,-step,step]
    for x in options:
        for y in options:
            img=image[location[1]+x:location[1]+s1+x,location[0]+y:location[0]+s2+y]
            score=computeScore(img,template)
            if score<best[2]:
                best=[(location[0]+y,location[1]+x),(location[0]+y+s2,location[1]+x+s1),score]
    return best[0],best[1]
            
    
def computeScore(img1,img2):
    err=np.sum((img1.astype("float") - img2.astype("float")) ** 2)
    err /= float(img1.shape[0] * img2.shape[1])
    return err
def computeScore2(img1,img2):
    hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
    hist1 = cv2.calcHist([hsv],[0,1,2],None,[HIST_SIZE,HIST_SIZE,HIST_SIZE],[0,256,0,256,0,256])
    hist2 = cv2.calcHist([hsv2],[0,1,2],None,[HIST_SIZE,HIST_SIZE,HIST_SIZE],[0,256,0,256,0,256])
    hist1 = cv2.normalize(hist1).flatten()
    hist2 = cv2.normalize(hist2).flatten()
    d=cv2.compareHist( hist1,hist2, cv2.cv.CV_COMP_CHISQR)
    return d

#check if object still resemble with original histogram
def check(frame,loc,original_hist):   
    start,stop=loc
    cup=frame[start[1]:stop[1],start[0]:stop[0]]
    hsv = cv2.cvtColor(cup, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv],[0,1,2],None,[HIST_SIZE,HIST_SIZE,HIST_SIZE],[0,256,0,256,0,256])
    hist = cv2.normalize(hist).flatten()
    d=cv2.compareHist( hist,original_hist, cv2.cv.CV_COMP_CHISQR)
    res= d>HIST_THRESH
    if res:
        print(d)
    return res

def addhist(img,loc):
    start,stop=loc
    cup=img[start[1]:stop[1],start[0]:stop[0]]
    hsv = cv2.cvtColor(cup, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv],[0,1,2],None,[HIST_SIZE,HIST_SIZE,HIST_SIZE],[0,256,0,256,0,256])
    hist = cv2.normalize(hist).flatten()
    histograms.append(hist)
    return
    
for loc in objects:
    addhist(previous,loc)

while(True):
    ret,frame=cap.read()
    now=cv2.resize(frame, (200,130), fx=0.5, fy=0.5)
 #   now=cv2.blur(now,(5,5))
    for i in range(0,len(objects)):
        loc=objects[i]
 #       previous=rotatedRect(previous,loc,5)
        cv2.rectangle(previous, loc[0],loc[1], (0,255,255), 1)
        objects[i]=trackObject(loc,previous,now)
    objects = [x for x in objects if not check(now,x,histograms[objects.index(x)])]
    cv2.imshow('frame',previous)
    k=cv2.waitKey(2)
    if k==32:
        cv2.waitKey(-1)
    if k==13:
        break
    previous=now
cap.release()
cv2.destroyAllWindows()
