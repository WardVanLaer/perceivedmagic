
import cv2
import numpy as np

#used variables
lowerskinHSV1 = np.array([0,30,80],np.uint8)
upperskinHSV1 = np.array([20,255,255],np.uint8)

lowerskinHSV2 = np.array([170,30,80],np.uint8)
upperskinHSV2 = np.array([180,255,255],np.uint8)

lowerskinBGR = np.array([20,40,95],np.uint8)
upperskinBGR = np.array([255,255,255],np.uint8)


lowerskinYCC = np.array([80,140,85],np.uint8)
upperskinYCC = np.array([255,180,135],np.uint8)
hands=[(((0,0),0,"CLOSED"),0),(((200,200),0,"CLOSED"),1)]

def reset():
    global hands
    hands=[(((0,0),0,"CLOSED"),0),(((200,200),0,"CLOSED"),1)]
def skinColor(frame):
    frame=cv2.medianBlur(frame,3)
    HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    Ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
    mask1 = np.uint8(cv2.inRange(HSV, lowerskinHSV1, upperskinHSV1) | cv2.inRange(HSV, lowerskinHSV2, upperskinHSV2))
    mask2 = cv2.inRange(Ycrcb, lowerskinYCC, upperskinYCC)
    bgr1 = cv2.inRange(frame, lowerskinBGR, upperskinBGR)
    bgr2=((np.amax(frame,axis=2)-np.amin(frame,axis=2))>15)*255
    bgr3= ((frame[:,:,0]<frame[:,:,2]) & (frame[:,:,1]<frame[:,:,2]))*255
    bgr4= (np.abs(frame[:,:,1]-frame[:,:,2])>15)*255
    mask3=np.uint8(bgr3 & bgr2 & bgr1 & bgr4)
    mask1=cv2.medianBlur(mask1,3)
    mask2=cv2.medianBlur(mask2,3)
    mask3=cv2.medianBlur(mask3,3)
    result=cv2.erode(mask1 & mask2 & mask3,None,3)
    result=cv2.dilate(result,None,3)
    result=cv2.medianBlur(result,3)
  #  cv2.imshow('mask1',mask1)
  #  cv2.imshow('mask2',mask2)
  # cv2.imshow('mask3',mask3)
    cv2.imshow('mask',result)
    return result

def show(frame):
    cv2.imshow('show',frame)
    k=cv2.waitKey(0)
    return
def tagger(hands,newlocs):
    newhands=[]
    if len(newlocs)>=2:
        if newlocs[0][0][0]<newlocs[1][0][0]:
            newhands.append((newlocs[0],0))
            newhands.append((newlocs[1],1))
        else:
            newhands.append((newlocs[0],1))
            newhands.append((newlocs[1],0))
    else:
        return tagger2(hands,newlocs)   
    return newhands
def tagger2(hands,newlocs):
    newhands=[]
    while(len(newlocs)!=0 and len(newhands)<2):
        min=1000000
        for h in hands:
            for n in newlocs:
                if (dist(n[0],h[0][0])<min ):
                    min=dist(n[0],h[0][0])
                    optimal=(n,h[1])
                    hand=h
        newhands.append(optimal)
        del newlocs[newlocs.index(optimal[0])]
        del hands[hands.index(hand)]
    return newhands+hands    
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

def extract_features(frame):
    global hands
    mask=skinColor(frame)
    (cnts, cmask) = cv2.findContours(mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    locs=[]
    areas=[]
    drawing = frame
    for cnt in cnts:
        (x, y, w, h) = cv2.boundingRect(cnt)
        if(w*h>500):
            areas.append(w*h)
            hull = cv2.convexHull(cnt)
            defects = cv2.convexityDefects(cnt,cv2.convexHull(cnt,returnPoints = False))
            centerin,radiusin=maxInscribedCircle(cnt)
            (x,y),radiusout = cv2.minEnclosingCircle(cnt)
            centerout=(int(x),int(y))
            defectscnt=0
          #  cv2.circle(drawing,centerin,int(radiusin),[255,0,255],1)
          #  cv2.circle(drawing,centerin,2,[255,0,255],-1)
            for i in range(defects.shape[0]):
                s,e,f,d = defects[i,0]
                start = tuple(cnt[s][0])
                end = tuple(cnt[e][0])
                far = tuple(cnt[f][0])
                if((dist(start,far)+dist(far,end))/2 >radiusin and (dist(start,far)+dist(far,end))/2<radiusout):
                 #   cv2.line(drawing,start,far,[0,0,255],1)
                 #   cv2.circle(drawing,far,2,[255,0,0],-1)
                    defectscnt+=1          
           # print(radiusout/dist(centerout,centerin),defectscnt)
            loc=(centerout,radiusout,"CLOSED")
            if((dist(centerout,centerin)*2.5>radiusout and defectscnt>4)):
             #   print("OPEN HAND "+str(defectscnt))
                loc=(centerout,radiusout,"OPEN")
                cv2.circle(drawing,centerout,10,[0,0,255],-1)
            locs.append(loc)
    hands=tagger(hands,locs)
    for h in hands:
        if h[1]==0:
            color=[0,255,255]
        else:
            color=[255,0,0]
        cv2.circle(drawing,h[0][0],int(h[0][1]),color,2)
        cv2.circle(drawing,h[0][0],2,color,-1)  
        
          
  #  show(drawing)
    return hands

def dist(a,b):
    return np.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)

def draw_locs(frame,loc):
    frame=cv2.resize(frame, (200,200))
    for loc in locs:
        print("DRAWING")
        cv2.rectangle(frame, (loc[0],loc[1]),(loc[2],loc[3]), (255, 255, 0), 1)
    return frame
def resize(frame,width):
    h,w,d=frame.shape
    frame=cv2.resize(frame,(width,int(width*h/w)))
    return frame

def handEvent(hands,events,time):
    if (abs(hands[0][0][0][0]-hands[1][0][0][0])+abs(hands[0][0][0][1]-hands[1][0][0][1]))>(hands[0][0][1]+hands[1][0][1]):
        handnames=["R","L"]
        for h,t in hands:
            if h[2]=="OPEN":   
                events[0].append("E4"+handnames[t])
                events[1].append(time)
    return events

"""
cap= cv2.VideoCapture('vids/hands.mov')
ret,frame=cap.read()


while True:
    ret,frame=cap.read()
    frame=resize(frame,300)
    cv2.imshow('source',frame)
    locs=extract_features(frame)
    k=cv2.waitKey(5)   
    if k==13:
        break
"""