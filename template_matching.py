import numpy as np
import cv2
from os import listdir
import math
from matplotlib import pyplot as plt
import imutils
from os.path import isfile, join



def match(img,temp,threshold):
    t1,t2=temp.shape
    result = cv2.matchTemplate(img, temp, cv2.TM_CCOEFF_NORMED)
    goodmatch=np.where(result>threshold)
    locs=[]
    #remove close matches
    for pt in zip(*goodmatch[::-1]):
        good=True
        for l in locs:
            if abs(l[0]-pt[0])<4 or abs(l[1]-pt[1])<4:
                good=False
        if good:
            locs.append(pt)
    return locs
    
def show(img):
    cv2.imshow('image',img)
    cv2.waitKey(0)          
        
def prepare(img):
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return img
def resize(img,height):
    x,y=img.shape[0:2]
    ratio=float(y)/float(x)
    img = cv2.resize(img, (int(height*ratio),height))
    return img

def start(template,image,size):   
    img=resize(image,size)
    template=prepare(template)
    prepared=prepare(img)
    mask=np.zeros((img.shape[0],img.shape[1]))
    for s in range(0,size/3,1):
        for angle in range(-5,5,2):
            temp=resize(template,size/2-s)
            temp = imutils.rotate(temp, angle)
            result=match(prepared,temp,0.75)
            t1,t2=temp.shape
            for pt in result:
                mask[pt[1]:pt[1]+t1,pt[0]:pt[0]+t2]+=np.ones((t1,t2))
    
                cv2.rectangle(img, pt, (pt[0] + t2, pt[1] + t1), (0,255,255), 1)             
 #   show(img)
    if np.max(mask)>0:
        mask=100*mask/np.max(mask)
    return mask
#vid
cap= cv2.VideoCapture('vid_hands.mov')
ret,image=cap.read()

show(image)
mypath='templates'
files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
#image = cv2.imread('img2.png')
size=60
mask=None
for f in files:
    if f[0]!='.':
        print(mypath+'/'+f)
        template = cv2.imread(mypath+'/'+f)
        if mask is None:
            mask=start(template,image,size)
        else:

            mask+=start(template,image,size)
mask=mask*255/np.max(mask)
show(np.uint8(mask))
threshold = cv2.inRange(mask,80,255)
show(threshold)

