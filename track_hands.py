import find_hands
import numpy as np
import cv2
from matplotlib import pyplot as plt


"""
location= (x,y,x+w,y+h)
"""



def show(frame):
    cv2.imshow('show',frame)
    k=cv2.waitKey(0)
    return

def draw_locs(frame,loc):
    frame=cv2.resize(frame, (200,200))
    for loc in locs:
        print("DRAWING")
        cv2.rectangle(frame, (loc[0],loc[1]),(loc[2],loc[3]), (255, 255, 0), 1)
    return frame

def extract_locs(mask):
    (cnts, cmask) = cv2.findContours(mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    locs=[]
    areas=[]
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        locs.append((x,y,x+w,y+h))
        areas.append(w*h)
    if len(areas)>1:
        indices=np.argsort(areas)[::-1]
        print(locs[indices[0]])
        return [locs[indices[0]],locs[indices[1]]]
    return locs

def update(new,old):
    hand1,hand2=old
    return old

cap= cv2.VideoCapture('vids/hands.mov')
ret,frame=cap.read()
HS=find_hands.handSubstractor(frame)
mask=HS.next(frame)
locs=extract_locs(mask)


while True:
    ret,frame=cap.read()
    mask=HS.next(frame)
    locs=extract_locs(mask) 
 #   hand_locs=update(locs,hand_locs)
    frame= draw_locs(frame,locs)
    cv2.imshow('source',frame)
    k=cv2.waitKey(5)   
    if k==13:
        break
