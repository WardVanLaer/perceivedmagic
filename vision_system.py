import numpy as np
import cv2
from matplotlib import pyplot as plt
import cups_colored as cc
import handfeatures as hf
import find_ball as fb
import progressbar
from time import sleep



def combineEvents(events):
    times=events[1]
    events=events[0]
    changes=True
    while changes:
        changes=False
        for i in range(len(events)-1):
            if(events[i]=='E4R' and events[i+1]=='E4L' and times[i+1]-times[i]<50):
                events[i]='E4L'
                events[i+1]='E4R'
                changes=True
    events2,times2=[],[]
    for i in range(len(events)-1):
        if events[i]!=events[i+1]:
            events2.append(events[i])
            times2.append(times[i])
    events2.append(events[-1])
    times2.append(times[-1])
    events,times=[],[]
    if not (events2[0][0]=='E' and ("IN"+events2[0][1] in events2[1])):
        events.append(events2[0])
        times.append(times2[0])
    for i in range(1,len(events2)-1):
        ev1,ev2,ev3=events2[i-1],events2[i],events2[i+1]
        if not ( (ev2[0]=='E' and ("IN"+ev2[1] in ev3)) or (ev2[0]=='E' and ("OUT"+ev2[1] in ev1))):
            events.append(ev2)
            times.append(times2[i])
    if not  (ev3[0]=='E' and ("OUT"+ev3[1] in ev2)):
        events.append(ev3)
        times.append(times2[i+1])
    return events,times

def resize(frame,width):
    h,w,d=frame.shape
    frame=cv2.resize(frame,(width,int(width*h/w)))
    return frame
def vision_system(path):
    events=[]
    events.append([])
    events.append([])
    cap= cv2.VideoCapture(path)
    frames=int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))-3
    if(path=="routines/v2.mov"):
        frames=1000
    count=5
    ret,frame=cap.read()
    hf.reset()
    fb.reset()
    cc.reset()
    frame=resize(frame,300)
    prevlocs=fb.get_locations(frame)
    prevcups=cc.extract_cups(frame)
    bar = progressbar.ProgressBar(maxval=frames, \
        widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    
    while count<frames:
        ret,frame=cap.read()
        ret,frame=cap.read()
        ret,frame=cap.read()
        bar.update(count)
        count+=3
        frame=resize(frame,300)
        locs=fb.get_locations(frame)
        cups=cc.extract_cups(frame)
        go=cc.cupMovements(prevcups,cups)
        hands=hf.extract_features(frame)
        events=hf.handEvent(hands,events,count)
        prevcups=cups
        if(not go):
            emptycup=cc.getEmptyCup()
            numprev=sum([l[4] for l in prevlocs])
            numnow=sum([l[4] for l in locs])
            if(len(prevlocs)!=len(locs) and numnow!=numprev):
                if(len(prevlocs)<len(locs)):
                    gone=fb.oddball(prevlocs,locs)
                    events=fb.ballEvent(gone,cups,hands,"OUT",events,frame,count)
                elif(len(prevlocs)>len(locs)):
                    gone=fb.oddball(locs,prevlocs)
                    events=fb.ballEvent(gone,cups,hands,"IN",events,frame,count)
                    fb.transferEvent(gone,cups,hands,events,count)
            elif emptycup>=0:
                events[0].append("E"+str(emptycup+1)+fb.getHandlingHand(cups,hands,emptycup))
                events[1].append(count)
            prevlocs=locs
            """
            cv2.imshow('source',frame)
            print(events)
            k=cv2.waitKey(0)
            """
    bar.finish()
    return combineEvents(events)

"""
cap= cv2.VideoCapture("routines/v2.mov")
count=0
frames=int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
while count<frames:
    print(count,frames)
    ret,frame=cap.read()
    ret,frame=cap.read()
    ret,frame=cap.read()
    ret,frame=cap.read()
    ret,frame=cap.read()
    count+=5
    frame=resize(frame,300)
    fb.show(frame)
"""