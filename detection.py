import numpy as np
import cv2
from matplotlib import pyplot as plt

#used variables
lowerskinHSV = np.array([0,50,100],np.uint8)
upperskinHSV = np.array([40,150,255],np.uint8)
lowercup = np.array([0,120,110],np.uint8)
uppercup = np.array([255,150,140],np.uint8)
lowercup2 = np.array([0,0,0],np.uint8)
uppercup2 = np.array([50,160,255],np.uint8)
lowercup3 = np.array([0,0,250],np.uint8)
uppercup3 = np.array([180,255,255],np.uint8)
lowerskinYCC = np.array([80,130,77],np.uint8)
upperskinYCC = np.array([255,180,135],np.uint8)
lowerball = np.array([0,200,180],np.uint8)
upperball = np.array([10,255,255],np.uint8)
skinbound=140
cap= cv2.VideoCapture('vid2.mov')
cap= cv2.VideoCapture('vid_hands.mov')
min_area=3000
HAND_AREA=0.01
thresh=0.5
pBalls=[]
rBalls=[]
#(history, nmixtures, backgroundRatio)
bgs= cv2.BackgroundSubtractorMOG()




def foreground(frame):
    fgmask=bgs.apply(frame,0.1)
    erosion = cv2.erode(fgmask,None,iterations = 1)
    dill = cv2.dilate(erosion,None,iterations = 5)
    return dill;
def movement(frame1,frame2):
    D=cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)-cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    mask=(np.absolute(D)>np.mean(frame1)*0.4)*1.0
    mask = cv2.erode(mask,None,iterations = 2)
    return mask


def ball(frame):
    # convert to hsv colorspace
    converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #remove small errors in background
    ballmask = cv2.inRange(converted, lowerball, upperball)
    ballmask = cv2.erode(ballmask,None,iterations = 1)
    #iterate over balls
    (cnts, cmask) = cv2.findContours(ballmask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        if w*h > 20 and w*h < 500:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 1)
    return frame
def hand(frame):
    # get skin mask  
    mask = skinColor(frame)
    #iterate over contours
    (cnts, cmask) = cv2.findContours(mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        bound=float(w*h)/np.size(mask)
        print(bound)
        if bound> HAND_AREA:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 1)

    return frame
                 

    return skinmask
def skinColor(frame):
    frame=cv2.blur(frame,(20,20))
    HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    """
    #method 1
    
    h,s,v=cv2.split(HSV)
    T=(s+v)/h
    mask=((T>5) & (T<25))
    skin1=np.zeros((frame.shape[0],frame.shape[1]))
    skin1[mask]=1
    """
    #method 2
    skin2 = cv2.inRange(HSV, lowerskinHSV, upperskinHSV)
    #method 3
    Ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
    skin3 = cv2.inRange(Ycrcb, lowerskinYCC, upperskinYCC)
    """
    cv2.imshow('frame2',skin2)
    cv2.imshow('frame3',skin3)
    """
    s=np.uint8((skin3/2+skin2/2))
    erode = cv2.erode(s,None,iterations = 4)
    dill  = cv2.dilate(erode,None,iterations =4)    
    dill=cv2.inRange(dill,150,256)
    return dill

def skinProb(frame):
    return frame


def normalize(frame):
    lab=cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
#-----Merge the CLAHE enhanced L-channel with the a and b channel-----------
    limg = cv2.merge((cl,a,b))
#-----Converting image from LAB Color model to RGB model--------------------
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final



#iterate over all frames of the video

ret,previous=cap.read()
previous=cv2.resize(previous, (400,260), fx=0.5, fy=0.5)
p=[previous]*2
i=0

while(True):
    ret,frame=cap.read()
    now=cv2.resize(frame, (400,260), fx=0.5, fy=0.5)
    fg=foreground(now)
    balls=ball(now)
    hands=hand(now)
    m=movement(now,p[i])
 #   cv2.imshow('frame',balls)
    """
    hands  = cv2.dilate(m*skin,None,iterations = 3)
    edges=cv2.Canny(now,100,200)
    dill  = cv2.dilate(edges,None,iterations = 2)
    """
    cv2.imshow('frame',hands)
    k=cv2.waitKey(5)
    if k==32:
        cv2.waitKey(-1)
    if k==13:
        break
    p[i]=now
    i=(i+1)%2
cap.release()
cv2.destroyAllWindows()



