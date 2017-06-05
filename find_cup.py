
import numpy as np
import cv2
from matplotlib import pyplot as plt

lowercup = np.array([30,0,60],np.uint8)
uppercup = np.array([90,255,255],np.uint8)
colors=[(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255),(255,0,255),(255,255,255)]
lasttag=0
offscreen=[]
ontop=[0]*3

def newTag(cups):
    global ontop
    loc1,tag1=cups[0]
    loc2,tag2=cups[1]
    x1,y1,x2,y2=loc1
    x3,y3,x4,y4=loc2
    #sideways
    if abs(x1-x3)>abs(y1-y3):
        if (y2-y1)<(y4-y3):
            return [tag1,tag2,"occ"]
        else:
            return [tag2,tag1,"occ"]
    elif tag1[-1]=="stacked":
        t1,t2=tag1[0:2]
        if y1>y3:         
            ontop[t2[0]]=tag2[0]+1
            return [t1,t2,tag2,"stacked"]
        else:
            ontop[tag2[0]]=t1[0]+1
            return [tag2,t1,t2,"stacked"]
    elif tag2[-1]=="stacked":
        t1,t2=tag2[0:2]
        if y1>y3:
            ontop[tag1[0]]=t1[0]+1
            return [tag1,t1,t2,"stacked"]
        else:
            ontop[t2[0]]=tag1[0]+1
            return [t1,t2,tag1,"stacked"]
        
    #above
    else:
        if y1>y3:
            ontop[tag1[0]]=tag2[0]+1
            return [tag1,tag2,"stacked"]
        else:
            ontop[tag2[0]]=tag1[0]+1
            return [tag2[0],tag1,"stacked"]
        
            
    #on top
def gettag(locs,oldtag):
    global ontop
    x1,y1,x2,y2=locs[0]
    x3,y3,x4,y4=locs[1]
    if len(oldtag)>3 and oldtag[-1]=="stacked":
        if ((y2-y1)>(y4-y3) and y1<y3) or ((y2-y1)<(y4-y3) and y1>y3):
            alone=oldtag[0]
            ontop[alone[0]]=0
        else:
            alone=oldtag[2]
            ison=oldtag[1]
            ontop[ison[0]]=0
        oldtag.remove(alone)
        
        if (y2-y1)>(y4-y3):
            return (oldtag,alone)
        return (alone,oldtag)
    if abs(x1-x3)>abs(y1-y3) and (y2-y1)>(y4-y3):
        return (oldtag[1],oldtag[0])
    if abs(x1-x3)<abs(y1-y3):
        t1,t2=oldtag[0:2]
        if y1<y3:
            ontop[t2[0]]=0
            return (oldtag[1],oldtag[0])
        else:
            ontop[t1[0]]=0
    return (oldtag[0],oldtag[1])
    
    
        

def fixOcclusion(cups,newlocs):
    t=20
    for loc in newlocs:
        x,y,x2,y2=loc
        occluded=[]
        for cup in cups:
            l,tag=cup
            a,b,a2,b2 =l
            scores=[abs(a-x)+abs(b-y),abs(a2-x2)+abs(b2-y2),abs(a2-x2)+abs(b-y),abs(a-x)+abs(b2-y2)]
            if min(scores)<t:
                occluded.append(cup)
        if len(occluded)>1:
            newlocs.remove(loc)
            result=tag_cups([c for c in cups if c not in occluded],newlocs)
            result.append((loc,newTag(occluded)))
            return result
    return tag_cups(cups,newlocs)

def fixAppearance(cups,newlocs):
    t=20
    global offscreen
    for cup in cups:
        loc,tag=cup
        if len(tag)>1:
            x,y,x2,y2=loc
            scores=[min(abs(a-x)+abs(b-y),abs(a2-x2)+abs(b-y),abs(a2-x2)+abs(b2-y2),abs(a-x)+abs(b2-y2)) for a,b,a2,b2 in newlocs]
            s=sorted(scores)         
            if s[1]<t :
                i=scores.index(s[0])
                scores[i]+=t
                j=scores.index(s[1])
                occluded=[newlocs[i],newlocs[j]]
                cups.remove(cup)
                result=tag_cups(cups,[l for l in newlocs if l not in occluded])
                tags=gettag(occluded,tag)
                for loc,t in zip(occluded,tags):
                    result.append((loc,t))
                return result
    result=cups+offscreen
    offscreen=[]
    return tag_cups(result,newlocs)



def update_locations(cups,newlocs):
    global ontop
    if len(cups)==0:
        return tag_cups(cups,newlocs) 
    #objects are occluded
    elif len(cups)>len(newlocs):
        return fixOcclusion(cups,newlocs)
    elif len(cups)<len(newlocs):
        return fixAppearance(cups,newlocs)
    else:
        return tag_cups(cups,newlocs)                

#find closest cup and add appropriate tag
def tag_cups(oldcups,locs):
    global lasttag,offscreen
    cups=[]
    if len(oldcups)==0:
        for l in locs:
            cups.append((l,[locs.index(l)]))
        lasttag=len(locs)-1    
        return cups
    for l in locs:
        dist=[abs(loc[0]-l[0])+abs(loc[1]-l[1]) for loc,tag in oldcups]
        tags=[tag for loc,tag in oldcups]
        if len(dist)>0:
            cup,tag=oldcups[dist.index(min(dist))]
            dist2=[abs(loc[0]-cup[0])+abs(loc[1]-cup[1]) for loc in locs]
            if locs[dist2.index(min(dist2))]==l:
                index=dist.index(min(dist))
                cup=(l,tags[index])
                cups.append(cup)
                oldcups.remove(oldcups[index])
            else:
                cups.append((l,[lasttag+1]))
                lasttag+=1
        else:
            cups.append((l,[lasttag+1]))
            lasttag+=1       
    offscreen+=oldcups           
    return cups
        

def show(frame):
    cv2.imshow('show',frame)
    cv2.waitKey(0)



def cup_mask(frame):
    # convert to hsv colorspace
   # show(frame)
    converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #remove small errors in background
    mask = cv2.inRange(converted, lowercup, uppercup)
   # show(mask)
    mask=cv2.dilate(mask,None,2)
    mask=cv2.erode(mask,None,4)
    mask = cv2.medianBlur(mask,5)
  #  show(mask)
    return mask

def find_cup(frame):
    mask=cup_mask(frame)
    cmask=np.copy(mask)
    (cnts, cmask) = cv2.findContours(cmask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    locs=[]
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        if float(w*h)>np.size(mask)*0.01 :
            locs.append((x,y,x+w,y+h))
    return locs

class Locator:
    def __init__(self):
        self.cups=[]
    def getdata(self):
        global ison
        return ison
    def update(self,frame):
        frame=cv2.resize(frame,(200,200))
        locs=find_cup(frame)
        self.cups=update_locations(self.cups,locs)
        return self.cups
    def draw_cups(self,frame,inside):
        frame=cv2.resize(frame, (200,200))
        for cup in self.cups:
            loc,tag=cup
            if len(tag)==1:
                color=colors[tag[0]]
            else:
                color=(0,0,0)
                
                tag=tag[0]
            font = cv2.FONT_HERSHEY_SIMPLEX
            if tag[0]>2:
                show(cup_mask(frame))
            incup=inside[tag[0]]
            if incup==-1:
                incup='?'
            cv2.putText(frame,str(incup),(loc[0]+5,loc[1]+20), font, 0.5,(0,0,255),2)
            cv2.rectangle(frame, (loc[0],loc[1]),(loc[2],loc[3]), color, 2)
        return frame
        
        
        

