import PFA
import numpy as np
import itertools
import progressbar
from time import sleep

events=['1IN1R','2IN1R','3IN1R','1IN1L','2IN1L','3IN1L','1IN2R','2IN2R','3IN2R','1IN2L','2IN2L','3IN2L','1IN3R','2IN3R','3IN3R','1IN3L','2IN3L','3IN3L','1OUT1R','2OUT1R','3OUT1R','1OUT1L','2OUT1L','3OUT1L','1OUT2R','2OUT2R','3OUT2R','1OUT2L','2OUT2L','3OUT2L','1OUT3R','2OUT3R','3OUT3R','1OUT3L','2OUT3L','3OUT3L','E1R','E1L','E2R','E2L','E3R','E3L','E4R','E4L','IN4R','IN4L','OUT4R','OUT4L']
states=[c1+c2+c3+R+L for R in ["0","1"] for L in ["0","1"] for c3 in ["0","1","2","3"] for c2 in ["0","1","2","3"] for c1 in ["0","1","2","3"]]


class logic:

    def __init__(self):
        self.pfa=PFA.model(256,48)
        self.fill()
    def getscore(self,input):
        return self.pfa.viterbi_decode(input)
    def fill(self):
        correction=10**(-10)
        difficult=0.006
        easy=0.03
        moves=["in","out","empty"]
        objects=[0,1,2]
        balls=[1,2,3]
        hands=[3,4]
        matrix_normal=np.identity(8)
        zero=np.zeros((8,8))
        index=0
        for cup in objects:
            for hand in hands:
                   for ball in balls:
                    self.pfa.setTransmat(self.inmatrix(cup,ball,hand,easy,difficult)+correction,index)
                    index+=1
        for cup in objects:
            for hand in hands:
                   for ball in balls:
                    self.pfa.setTransmat(self.outmatrix(cup,ball,hand,easy,difficult)+correction,index)
                    index+=1
        for cup in objects:
            for hand in hands:
                self.pfa.setTransmat(self.emptymatrix(cup,hand,easy,difficult)+correction,index)
                index+=1
        for hand in hands:
            self.pfa.setTransmat(self.emptyhand(hand,easy)+correction,index)
            index+=1
        for hand in hands:
            self.pfa.setTransmat(self.inhand(hand,easy,difficult)+correction,index)
            index+=1
        for hand in hands:
            self.pfa.setTransmat(self.outhand(hand,easy,difficult)+correction,index)
            index+=1
   #     self.pfa.normalize()           
                
    def inmatrix(self,cup,ball,hand,p_easy,p_dificult):
        matrix=np.zeros((len(states),len(states)))
        for i in states:
            for j in states:           
                if(all([i[k]==j[k] for k in range(len(i)) if (k!=cup and k!=hand)])):
                    if(i[cup]=="0" and i[hand]=="0" and j[cup]==str(ball-1) and j[hand]=="1"):
                        matrix[states.index(i),states.index(j)]=p_dificult
                    elif(i[cup]=="0" and i[hand]=="0" and j[cup]==str(ball) and j[hand]=="0"):
                        matrix[states.index(i),states.index(j)]=p_easy
                    elif(i[cup]=="0" and i[hand]=="1" and j[cup]==str(ball) and j[hand]=="1"):
                        matrix[states.index(i),states.index(j)]=p_easy
                    elif(i[cup]=="0" and i[hand]=="1" and j[cup]==str(ball+1) and j[hand]=="0"):
                        matrix[states.index(i),states.index(j)]=p_dificult
                    elif(j[cup]==str(int(i[cup])+ball) and i[hand]==j[hand]):
                        matrix[states.index(i),states.index(j)]=p_dificult
        return matrix
    
    def outmatrix(self,cup,ball,hand,p_easy,p_dificult):
        matrix=np.zeros((len(states),len(states)))
        for i in states:
            for j in states:
                if(all([i[k]==j[k] for k in range(len(i)) if (k!=cup and k!=hand)])):
                    if(i[cup]==str(ball-1) and i[hand]=="1" and j[cup]=="0" and j[hand]=="0"):
                        matrix[states.index(i),states.index(j)]=p_dificult
                    elif(i[cup]==str(ball) and i[hand]=="1" and j[cup]=="0" and j[hand]=="1"):
                        matrix[states.index(i),states.index(j)]=p_easy
                    elif(i[cup]==str(ball) and i[hand]=="0" and j[cup]=="0" and j[hand]=="0"):
                        matrix[states.index(i),states.index(j)]=p_easy
                    elif(i[cup]==str(ball) and i[hand]=="1" and j[cup]=="1" and j[hand]=="0"):
                        matrix[states.index(i),states.index(j)]=p_dificult
        return matrix
    
    def emptymatrix(self,cup,hand,p_easy,p_dificult):
        matrix=np.zeros((len(states),len(states)))
        for i in states:
            for j in states:
                if(all([i[k]==j[k] for k in range(len(i)) if (k!=cup and k!=hand)])):
                    if(i[cup]=="0" and j[cup]=="0" and j[hand]==i[hand]):
                        matrix[states.index(i),states.index(j)]=p_easy
                    elif(i[cup]==j[cup] and j[hand]=="0" and i[hand]=="0"):
                        matrix[states.index(i),states.index(j)]=p_dificult
                    elif(i[cup]=="1" and j[cup]=="0" and j[hand]=="1" and i[hand]=="0"):
                        matrix[states.index(i),states.index(j)]=p_dificult
                    elif(i[cup]=="0" and j[cup]=="1" and j[hand]=="0" and i[hand]=="1"):
                        matrix[states.index(i),states.index(j)]=p_dificult
        return matrix
    def emptyhand(self,hand,p_easy):
        matrix=np.zeros((len(states),len(states)))
        for i in states:
            for j in states:
                if(all([i[k]==j[k] for k in range(len(i))]) and i[hand]=="0"):
                    matrix[states.index(i),states.index(j)]=p_easy
        return matrix
    def inhand(self,hand,p_easy,p_dificult):
        matrix=np.zeros((len(states),len(states)))
        for i in states:
            for j in states:
                if(all([i[k]==j[k] for k in range(len(i)-2)])):
                    if(hand==3 and i[3]=="0" and j[3]=="1" and i[4]==j[4]):
                        matrix[states.index(i),states.index(j)]=p_easy
                    if(hand==4 and i[4]=="0" and j[4]=="1" and i[3]==j[3]):
                        matrix[states.index(i),states.index(j)]=p_easy
                    if(hand==3 and i[3]=="0" and j[3]=="0" and i[4]=="0" and j[4]=="1"):
                        matrix[states.index(i),states.index(j)]=p_dificult
                    if(hand==4 and i[4]=="0" and j[4]=="0" and i[3]=="0" and j[3]=="1"):
                        matrix[states.index(i),states.index(j)]=p_dificult
        return matrix
    def outhand(self,hand,p_easy,p_dificult):
        matrix=np.zeros((len(states),len(states)))
        for i in states:
            for j in states:
                if(all([i[k]==j[k] for k in range(len(i)-2) if(k!=hand)])):
                    if(i[hand]=="1" and j[hand]=="0"):
                        matrix[states.index(i),states.index(j)]=p_easy
        return matrix
                    
    def printresult(self,result):
        for r in result:
            print(self.translate(r))
    def translate(self,list):
        print(list)
        translation=[]
        for i in list:
            translation.append(states[i])    
        return translation                

    def analyze(self,input):
        print("------running viterbi------")
        bar = progressbar.ProgressBar(maxval=len(input[0]), \
            widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        s=[]
        MOI=[]
        for e in input[0]:
            s.append(events.index(e))
        result=self.getscore(s[:1])
      #  self.printresult(result[0])
        p=result[2]
        for i in range(1,len(s)):
            bar.update(i)
            t=s[:i+1]
            result=self.getscore( t)
          #  self.printresult(result[0])
            p_prefix=result[2]
            p_now=result[1]
            score1=p/p_prefix
            score2=p/p_now
           # print(score1,score2)
            if score2>40 or score1>1.5:
                MOI.append(input[1][i])
            p=p_now
        bar.finish()
        return MOI
   
"""
test=[['E3R','1OUT1L','1IN1L','1OUT3L','1IN3L','E1R','1IN1L','1OUT3L'],[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]]
l=logic()
l.analyze(test)

"""
