


import cv2
import numpy as np

#blue cup
cup1HSVl= np.array([80,180,150],np.uint8)
cup1HSVh = np.array([110,255,255],np.uint8)
#green cup
cup2HSVl= np.array([40,70,80],np.uint8)
cup2HSVh = np.array([80,220,255],np.uint8)

#red cup
cup3HSVl= np.array([0,125,50],np.uint8)
cup3HSVh = np.array([6,205,215],np.uint8)
cup3HSVl2= np.array([172,125,50],np.uint8)
cup3HSVh2 = np.array([180,220,215],np.uint8)
filters=[(cup1HSVl, cup1HSVh),(cup2HSVl, cup2HSVh),(cup3HSVl, cup3HSVh,cup3HSVl2, cup3HSVh2)]
colors=[(255,255,0),(0,255,0),(255,0,255)]
movingdown=-1
movingup=-1
empty=-1
countup,countdown=0,0
cupslocal=[((-1,-1),0),((-1,-1),1),((-1,-1),2)]
stacked=[False,False,False]
movement=[0,0,0]
def reset():
    global movingdown,movingup,empty,countup,countdown,cupslocal,stacked
    movingdown=-1
    movingup=-1
    empty=-1
    countup,countdown=0,0
    cupslocal=[((-1,-1),0),((-1,-1),1),((-1,-1),2)]
    stacked=[False,False,False]
    
def extract_cups(frame):
    global cupslocal
  #  frame=cv2.medianBlur(frame,3)
    cups=[]
    i=0;
    for filter in filters:
        found=False
        cup=((0,0),(0,0))
        HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = np.uint8(cv2.inRange(HSV, filter[0], filter[1]))
        if(len(filter)>2):
            mask = mask | np.uint8(cv2.inRange(HSV, filter[2], filter[3]))
        mask=cv2.erode(mask,None,3)
        mask=cv2.dilate(mask,None,5)
        mask=cv2.medianBlur(mask,5)
        if(i==2):
            cv2.imshow('cup',mask)
        (cnts, cmask) = cv2.findContours(mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        size=0
        for cnt in cnts:
            (x, y, w, h) = cv2.boundingRect(cnt)
            area=cv2.contourArea(cnt)
            if(size<area):
                size=area
                cup=(x+w/2,y+h/2)
                found=True
       # print(size/np.size(mask))
        if(found and size> (0.001*np.size(mask))):
            cv2.circle(frame,cup,4,(255,0,0),-1)
            cupslocal[i]=cup,i
            cups.append((cup,i))
        else:
            cups.append(cupslocal[i])
        i+=1
    cv2.imshow('cups',frame)
    return cups


def resize(frame,width):
    h,w,d=frame.shape
    frame=cv2.resize(frame,(width,int(width*h/w)))
    return frame
def getStacked(cups):
    stacked=[False,False,False]
    for c,t in cups:
        for cc,tt in cups:
            if (tt!=t and c[1]-cc[1]<30 and c[1]-cc[1]>0 and abs(c[0]-cc[0])<20):
                stacked[tt]=True
    return stacked
                
        
        
def cupMovements(previous,now):
    stacked=getStacked(now)
    foundup,founddown=False,False
    global movingdown,countup,countdown,movingup,empty,movement
    for cupp,tagp in previous:
        for cupn,tagn in now:
            if(tagn==tagp and cupp[0]!=-1 and not stacked[tagn]):
               # print(abs(cupp[1]-cupn[1])+abs(cupp[0]-cupn[0]))
                if(abs(cupp[1]-cupn[1])+abs(cupp[0]-cupn[0])<70):
                    movement[tagn]=cupp[1]-cupn[1]
                else:
                    movement[tagn]=0
                if(movement[tagn]>6 ):
                    countup=0
                    foundup=True
                    movingup=tagn
                if(movement[tagn]<-6):
                    countdown=0
                    founddown=True
                    movingdown=tagn
   
    if not foundup:
        if(countup==5):
            movingup=-1
        if(countup==0):
            empty=movingup
        countup+=1      
    if not founddown:
        countdown+=1
        if(countdown>3):
            movingdown=-1
   # print(movingup,movingdown,empty)
   # print(movement)
    return foundup or founddown

def getEmptyCup():
    global empty
    r=empty
    empty=-1
    return r
def getLastMoving():
    if movingdown !=-1:
        return movingdown
    else:
        return movingup
def getLastMovingDown():
    return movingdown

    

            
"""        
cap= cv2.VideoCapture('vids/colored3.mov')
ret,frame=cap.read()
frame=resize(frame, 200)
prev=extract_cups(frame)

while True:
    print("loop")
    ret,frame=cap.read()
    ret,frame=cap.read()
    ret,frame=cap.read()
    frame=resize(frame, 200)
    cv2.imshow('source',frame)
    cups=extract_cups(frame)
    cupMovements(prev,cups)
    prev=cups
    cv2.imshow('source',frame)
    k=cv2.waitKey(0)   
    if k==13:
        break
"""