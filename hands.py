
import numpy as np
from scipy import misc
from matplotlib import pyplot as plt
import cv2


#used variables
lowerskinHSV1 = np.array([0,30,80],np.uint8)
upperskinHSV1 = np.array([15,255,255],np.uint8)

lowerskinHSV2 = np.array([170,30,80],np.uint8)
upperskinHSV2 = np.array([180,255,255],np.uint8)

lowerskinBGR = np.array([20,40,95],np.uint8)
upperskinBGR = np.array([255,255,255],np.uint8)


lowerskinYCC = np.array([80,135,85],np.uint8)
upperskinYCC = np.array([255,180,135],np.uint8)

def dist(a,b):
        return np.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)

class HandEvents():
    def __init__(self):
        self.bins=10
        self.load_bayes(self.bins)
    def apply(self,frame):
        self.extract_features(frame)
        return
        
    def load_bayes(self,bins):
        s=256/bins
        noskin=np.ones((bins,bins,bins))
        skin=np.zeros((bins,bins,bins))
        bayes=np.zeros((bins,bins,bins))
        count=0
        F = open("Skin_NonSkin.txt","r")
        for line in F:
            count+=1
            l=line[:len(line)-2].split("\t")
            i1=int(int(l[0])/s)-1
            i2=int(int(l[1])/s)-1
            i3=int(int(l[2])/s)-1
            if(l[3]=="1"):
                skin[i1,i2,i3]+=1
            if(l[3]=="2"):
                noskin[i1,i2,i3]+=1
        noskin=noskin
        skin=skin
        print("Loaded "+str(count)+" sample pixels")  
        self.bayes=(skin/np.sum(skin))/(noskin/np.sum(skin))
        return
    
    def extract_features(self,frame):
        mask_static=self.skinColor(frame)
        mask_bayes=self.apply_bayes(frame)
        mask=mask_static & mask_bayes
        show(mask_static)
        (cnts, cmask) = cv2.findContours(mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        locs=[]
        areas=[]
        for cnt in cnts:
            (x, y, w, h) = cv2.boundingRect(cnt)
            if(w*h>1000):
                locs.append((x,y,x+w,y+h))
                areas.append(w*h)
                hull = cv2.convexHull(cnt)
                defects = cv2.convexityDefects(cnt,cv2.convexHull(cnt,returnPoints = False))
                drawing = frame
              #  cv2.drawContours(drawing,[hull],0,(0,0,255),2)
                center=(0,0)
                for i in range(defects.shape[0]):
                    s,e,f,d = defects[i,0]
                    start = tuple(cnt[s][0])
                    end = tuple(cnt[e][0])
                    far = tuple(cnt[f][0])
                    center=tuple([center[i]+far[i] for i in range(0,2)])
                palm=tuple([a/defects.shape[0] for a in center])
                cv2.circle(drawing,palm,5,[255,255,255],-1)
                radiusin=200
                for i in range(defects.shape[0]):
                    s,e,f,d = defects[i,0]
                    far = tuple(cnt[f][0])
                    r=dist(palm,far)
                    if r<radiusin:
                        radiusin=r
                for i in range(defects.shape[0]):
                    s,e,f,d = defects[i,0]
                    start = tuple(cnt[s][0])
                    end = tuple(cnt[e][0])
                    far = tuple(cnt[f][0])
                    if((dist(start,far)+dist(far,end))/2 >radiusin):
                        cv2.line(drawing,start,far,[0,0,255],1)
                        cv2.circle(drawing,end,5,[255,0,0],-1)
                cv2.circle(drawing,palm,int(radiusin),[255,255,0],1)
                (x,y),radiusout = cv2.minEnclosingCircle(cnt)
                center = (int(x),int(y))
                cv2.circle(drawing,center,int(radiusout),[0,255,255],1)
                print(dist(center,palm),radiusout)
                if(dist(center,palm)*3>radiusout):
                    print("OPEN HAND")
                show(drawing)      
        return locs
    
    
    
    def apply_bayes(self,image):
        s=256/self.bins
        print("apply method")
        x,y,d=np.shape(image)
        mask=np.zeros((x,y))
        for i in range(0,x):
            for j in range(0,y):
                b,g,r=image[i,j]
                mask[i,j]=self.bayes[int(b/s)-1,int(g/s)-1,int(r/s)-1]        
        mask=(mask>0.7)*255;
        result=cv2.erode(np.uint8(mask),None,3)
        result=cv2.dilate(result,None,4) 
        result=cv2.medianBlur(result,3)
        print("bayes done")
        return result
    
    
    def skinColor(self,frame):
        HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        Ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
        mask1 = np.uint8(cv2.inRange(HSV, lowerskinHSV1, upperskinHSV1) | cv2.inRange(HSV, lowerskinHSV2, upperskinHSV2))
        mask2 = cv2.inRange(Ycrcb, lowerskinYCC, upperskinYCC)
        bgr1 = cv2.inRange(frame, lowerskinBGR, upperskinBGR)
        bgr2=((np.amax(frame,axis=2)-np.amin(frame,axis=2))>15)*255
        bgr3= ((frame[:,:,0]<frame[:,:,2]) & (frame[:,:,1]<frame[:,:,2]))*255
        bgr4= (np.abs(frame[:,:,1]-frame[:,:,2])>15)*255
        mask3=np.uint8(bgr3 & bgr2 & bgr1 & bgr4)
        result=cv2.dilate(mask1 & mask2 & mask3,None,3)
        result=cv2.medianBlur(result,3)
        return result

def show(img):
        cv2.imshow('show',img)
        cv2.waitKey(0)
        return
    

h=HandEvents()
frame=cv2.imread('test2.png')
frame=cv2.resize(frame,(100,100))
h.apply(frame)
cap= cv2.VideoCapture('vids/hands3.mov')


while True:
    ret,frame=cap.read()
    frame=cv2.resize(frame, (100,100))
    h.apply(frame)
    cv2.imshow('source',frame)
    k=cv2.waitKey(5)   
    if k==13:
        break

    
    
    
    