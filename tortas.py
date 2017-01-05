import cv2
import cv2.cv as cv
import numpy as np
import os
from matplotlib import pyplot as plt
from timeit import default_timer as timer
from os import listdir
from os.path import isfile, join


lower = np.array([0,0,70],np.uint8)
upper = np.array([255,255,255],np.uint8)
lowerburned = np.array([0,0,0],np.uint8)
upperburned = np.array([50,200,200],np.uint8)
lowercruda = np.array([240,0,0],np.uint8)
uppercruda = np.array([255,255,255],np.uint8)
lowerag = np.array([0,0,0],np.uint8)
upperag = np.array([255,255,40],np.uint8)

def openImage(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (0,0), fx=0.3, fy=0.3) 
    #   cv2.imshow('pie',img)
    #   cv2.waitKey(0)
    return img


def maskPie(pie):
    img = cv2.medianBlur(pie,5)
    mask = cv2.inRange(pie, lower,upper)
    return mask



def show(img):
    cv2.imshow('pie',img)
    cv2.waitKey(0)
    return
def getROI(img,mask):
    #border size
    b=0
    (cnts, cmask) = cv2.findContours(mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        if w*h >50:
 #           cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 1)      
            return img[y-b:y+h+b,x-b:x+w+b]
    return None

def detectDeformada(img):
    mask=maskPie(img)
    return np.sum(mask)<700000
    #threshold 0.1 gives good results
def detectQuemada(ROI,threshold):
    img = cv2.medianBlur(ROI,5)
    img=cv2.resize(ROI, (50,50))     
    mask = cv2.inRange(img, lowerburned,upperburned)
    mask = cv2.erode(mask,None,iterations = 1)
    result=np.mean(mask)/255
    return result>threshold

    #threshold 0,3 gives good results
def detectCruda(ROI,threshold):
    img=cv2.resize(ROI, (50,50))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    img = cv2.medianBlur(img,3)
    mask = cv2.inRange(img, lowercruda,uppercruda)
    result=np.sum(mask)/255
    return result>threshold*2500
    #threshold 0.3 gives good results
def detectAgu(ROI,threshold):
    img=cv2.resize(ROI, (100,100))
    mask = cv2.inRange(img, lowerag,upperag)
    mask = cv2.dilate(mask,None,iterations = 2)
    (cnts, cmask) = cv2.findContours(mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    return len(cnts)>4
#threshold 0.55
def detectRota(ROI,threshold):
    a,b,c=ROI.shape
    x=a/2
    y=b/2   
    cv2.circle(ROI,(x,y), x/2,(0,0,0),-1)
    ROI=maskPie(ROI)
    ROI= cv2.erode(ROI,None,iterations = 2)
    show(ROI)
    q1=ROI[0:x,0:y]
    q2=ROI[x:,0:y]
    q3=ROI[0:x,y:]
    q4=ROI[x:,y:]
    m=1
    for q in [q1,q2,q3,q4]:
        score=np.mean(q)/255
        if score<m:
            m=score
        if threshold>score:
            return True
    print(m)
    return False
def printText(img,text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img,text,(30,30), font, 1,(255,255,255),2)
    return img

def start(img):
    fault="Pie not faulty"
    start=timer()   
    mask=maskPie(img)
    ROI=getROI(img,mask)
    if detectDeformada(img):
        fault="Deformada"
    elif detectCruda(ROI,0.3):
        fault="Cruda"
    elif detectRota(ROI,0):
        fault="Rota"
    elif detectAgu(ROI,0.3):
        fault="Agujereada"
    elif detectQuemada(ROI,0.01):
        fault="Quemada"


    end=timer()
    printText(img,fault)
    time=end-start
    print("Pie was evaluated in "+str(time*1000)+" ms")
    print("Result: "+fault)

    
mypath='ImagenesTortas/TortasUnitarias/all'
files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
print("starting pie evaluation")
print("found "+str(len(files))+" pie images in folder")
for f in files:
    print("----------------------")
    print("file: "+f)
    img=openImage(mypath+"/"+f)
    start(img)


