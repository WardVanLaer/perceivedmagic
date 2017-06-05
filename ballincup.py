import numpy as np
import find_cup
import find_ball
import cv2
from matplotlib import pyplot as plt
IN,OUT,EMPTY=0,1,2

def show(frame):
    cv2.imshow('show',frame)
    cv2.waitKey(0)

class logic:
    def __init__(self,frame):
        self.balls=find_ball.get_locations(frame)
        self.cuplocator=find_cup.Locator()
        self.cups=self.cuplocator.update(frame)
        self.inside=[-1]*3
        self.cupsmem=np.zeros(10)
        self.moving=[False]*3
        self.ballevent=[False]*3
        self.history=[]
    def newEvent(self,event):
        if(len(self.history)==0):
            self.history.append(event)
        elif(event != self.history[-1]):
            self.history.append(event)
        print(self.history)
    def color(self,tag):
        colors=["BLUE","GREEN","RED"]
        return colors[tag]
    def update(self,frame):
        newballs=find_ball.get_locations(frame)
        newcups=self.cuplocator.update(frame)
       # print(newcups)
       # print(self.cups)
        self.emptyCup(newcups)
        #ball gone
        if len(newballs)!=len(self.balls):
          #  print("-------------------------------------")
            appear=True
            if len(newballs)<len(self.balls):
                gone=self.oddball(newballs,self.balls)
                appear=False
            else:
                gone=self.oddball(self.balls,newballs)
            for c,t in self.cups:
                newcup=[cup for cup,tag in newcups if t==tag]
                if len(newcup)==0 and len(t)>2:
                    t=t[1]
                    self.ballevent[t[0]]=True
                    print("COMES FROM CUP WITH TAG: "+self.color(t[0]))
                    self.newEvent(OUT+3*t[0])
                    self.inside[t[0]]=0
                elif len(newcup)==0:
                    x,y,x2,y2=c
                    if c[3]<gone[3] and c[0]<gone[0]<c[2]:
                        print("FOUND CUP WITH TAG: "+self.color(t[0]))
                    
                else:
                    x,y,x2,y2=newcup[0]
                    score1=(((gone[0]-x)**2+(gone[3]-y2)**2)**(0.5)+((gone[2]-x2)**2+(gone[3]-y2)**2)**(0.5))/2
                    score2=gone[3]-c[3]
                    score3=min(abs(c[1]-y),abs(c[3]-y2))
                    c=newcup[0]
                    print(score1,score2,score3)
                    if score1<20 and score2>0 and score3!=0:                
                        if len(t)>1:
                            t=t[0]
                        self.ballevent[t[0]]=True
                        if not appear:
                            print("FOUND CUP WITH TAG: "+self.color(t[0]))
                            self.newEvent(IN+3*t[0])
                            self.inside[t[0]]=1
                        if appear:
                            print("COMES FROM CUP WITH TAG: "+self.color(t[0]))
                            self.newEvent(OUT+3*t[0])
                            self.inside[t[0]]=0
        self.balls=newballs[:]
        self.cups=newcups[:]       
        return True
    def draw(self,frame):
        frame=find_ball.draw_locs(frame,self.balls)
        frame=self.cuplocator.draw_cups(frame,self.inside)
        return frame
    def oddball(self,balls,balls_more):
        if len(balls)==0:
            return balls_more[0]
        for a,b,c,d in balls_more:
                dist=min([abs(x-a)+abs(y-b) for x,y,x2,y2 in balls])
                if dist>8:
                    return a,b,c,d
    def emptyCup(self,newcups):
        threshold=3
        if len(self.cups)==len(newcups):
            for c,cuptag in self.cups:
                if cuptag[-1]!="occ":
                    t=cuptag[0]
                    if type(t) is list:
                        t=t[0]
                    x,y,x2,y2=c
                    score1=sum([min(abs(b-y),abs(d-y2)) for (a,b,c,d),tag in newcups if tag==cuptag])                  
                  #  print(score1,self.cupsmem[t],self.ballevent[t])
                    if self.cupsmem[t]>threshold:
                        self.moving[t]=True
                    elif not self.ballevent[t] and self.moving[t]:
                        self.moving[t]=False
                        print("EMPTY CUP "+self.color(t))
                        self.newEvent(EMPTY+3*t)
                        self.inside[t]=0
                    else:
                        self.moving[t]=False
                        self.ballevent[t]=False
                    self.cupsmem[t]=(max(score1,0)+self.cupsmem[t])/2
        return True            
cap= cv2.VideoCapture('vids/problem.mov')
ret,frame=cap.read()
frame=cv2.resize(frame,(200,200))
logic=logic(frame)
while True:
    ret,frame=cap.read()
    frame=cv2.resize(frame,(200,200))
    logic.update(frame)
    frame=logic.draw(frame)
    cv2.imshow('cups',frame)
    k=cv2.waitKey()   
    if k==13:
        break

    
