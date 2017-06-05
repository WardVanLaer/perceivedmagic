import numpy as np
import cv2
from matplotlib import pyplot as plt
import cups_colored as cc


lowerball = np.array([0,190,180],np.uint8)
upperball = np.array([10,255,255],np.uint8)
lowerball2 = np.array([170,190,180],np.uint8)
upperball2 = np.array([181,255,255],np.uint8)
calibrateArea=[0,0]

def reset():
    global calibrateArea
    calibrateArea=[0,0]
    
def show(frame):
    cv2.imshow('show',frame)
    cv2.waitKey(0)

def draw_ball_locs(frame,locs):
    for loc in locs:
        if loc[4]==1:
            cv2.rectangle(frame, (loc[0],loc[1]),(loc[2],loc[3]), (255, 255, 0), 1)
        elif loc[4]==2:
            cv2.rectangle(frame, (loc[0],loc[1]),(loc[2],loc[3]), (255, 0, 255), 1)
        else:
            cv2.rectangle(frame, (loc[0],loc[1]),(loc[2],loc[3]), (0, 255, 255), 1)
    cv2.imshow('balls',frame)
    return

def ball_mask(frame):
    # convert to hsv colorspace
    converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #remove small errors in background
    mask1 = cv2.inRange(converted, lowerball, upperball)
    mask2 = cv2.inRange(converted, lowerball2, upperball2)
    mask=mask1|mask2
    mask=cv2.erode(mask,None,5)
    mask=cv2.dilate(mask,None,2)
   # mask = cv2.medianBlur(mask,3)
    cv2.imshow('ballmask',mask)
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

def get_locations(frame):
    mask=ball_mask(frame)
    copymask=np.copy(mask)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    (cnts, cmask) = cv2.findContours(mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    locs=[]
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        score=cv2.contourArea(c)/(frame.shape[0]*frame.shape[1])
        if score>0.0005 and score<0.01:
            center,radius=maxInscribedCircle(c)
            if calibrateArea[1]>300:
                score=calibrateArea[1]*cv2.contourArea(c)/calibrateArea[0]
                t=[2.22,1.26]
            else:
                score=cv2.contourArea(c)/(3.14*radius**2)
                t=[2.5,1.87]
           # print(score)
            if( score> t[0] ):
                locs.append((x,y,x+w,y+h,3))
            elif( score>t[1]):
                locs.append((x,y,x+w,y+h,2))
            else:
                calibrateArea[0]+=cv2.contourArea(c)
                calibrateArea[1]+=1
                locs.append((x,y,x+w,y+h,1))

    draw_ball_locs(frame,locs)
    return locs
def dist(a,b):
    return np.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)
def resize(frame,width):
    h,w,d=frame.shape
    frame=cv2.resize(frame,(width,int(width*h/w)))
    return frame
def oddball(balls,ballsmore):
    balls_more=ballsmore[:]
    if len(balls)==0:
        return balls_more
    gone=[]
    while len(balls)!=len(balls_more):
        maxdist=0
        best=0
        for a,b,c,d,n in balls_more:
                dist=min([abs(x-a)+abs(y-b) for x,y,x2,y2,n2 in balls])
                if dist>maxdist:
                    maxdist=dist
                    best=(a,b,c,d,n)
        gone.append(best)
        del balls_more[balls_more.index(best)]
    return gone
def ballEvent(balls,cups,hands,inout,events,frame,time):
    #print("BALLEVENT")
    handnames=["R","L"]
    ballincup=[0,0,0]
    num=sum([ball[4] for ball in balls])
    num=min(num,3)
    tag=cc.getLastMoving()
   # print(tag)
    if (tag>0):
        dist=[abs(balls[0][0]-c[0]) for c,t in cups if t==tag]
        if(dist[0]>50):
            tag=-1
   # print(tag)
    
    if( tag<0):
        dist=[abs(balls[0][0]-c[0])+balls[0][1]-c[1] for c,t in cups]
      #  print(dist)
        if(min(dist)<20):
            tag=cups[dist.index(min(dist))][1]
         #   print(abs(cc.movement[tag]))
            if(abs(cc.movement[tag])<5):
                tag=-1
    if(tag>=0):
        events[0].append(str(num)+inout+str(tag+1)+getHandlingHand(cups,hands,tag))
        events[1].append(time)
    return events
def transferEvent(balls,cups,hands,events,time):
    if (max([balls[0][1]-c[1] for c,t in cups])<-50) and (abs(hands[1][0][0][0]-hands[0][0][0][0])+abs(hands[1][0][0][1]-hands[0][0][0][1])<80):
            events[1].append(time)
            events[0].append('IN4R')        
    return events
def getHandlingHand(cups,hands,tag):
    handnames=["R","L"]
    for c in cups:
        if c[1]==tag:
            cup=c[0]
    if((2*abs(cup[0]-hands[0][0][0][0])+abs(cup[1]-hands[0][0][0][1]))>(2*abs(cup[0]-hands[1][0][0][0])+abs(cup[1]-hands[1][0][0][1]))):
            return handnames[hands[1][1]]
    else:
        return handnames[hands[0][1]]



