import PFA
import numpy as np
import itertools

IN1L,OUT1L,E1L,IN1R,OUT1R,E1R,IN2L,OUT2L,E2L,IN2R,OUT2R,E2R,IN3L,OUT3L,E3L,IN3R,OUT3R,E3R,IN4L,OUT4L,E4L,IN4R,OUT4R,E4R=range(0,24)
states=["000", "100", "010", "110", "001", "101", "011", "111"]
hands=["","R","L","RL"]


class logic:

    def __init__(self):
        self.pfa=PFA.model(32,24)
        self.fill()
    def getscore(self,input):
        return self.pfa.viterbi_decode(input)
    def fill(self):
        difficult=0.02
        easy=0.05
        moves=["in","out","empty"]
        objects=[0,1,2]
        hands=["left","right"]
        matrix_normal=np.identity(8)
        zero=np.zeros((8,8))
        index=0
        for cup in objects:
            for hand in hands:
                for move in moves:
                    if move=="in":
                        normal,in_hand,out_hand=self.inmatrix_easy(cup)*easy,difficult*self.inmatrix_diff(cup),zero
                    if move=="out":
                        normal,in_hand,out_hand=self.outmatrix_easy(cup)*easy,zero,self.outmatrix_diff(cup)*difficult
                    if move=="empty":
                        normal,in_hand,out_hand=self.emptymatrix_easy(cup)*easy,self.emptymatrix_diff1(cup)*difficult,self.emptymatrix_diff2(cup)*difficult
                    if hand=="left":
                        self.pfa.setTransmat(np.bmat([[normal,zero,in_hand,zero],[zero,normal,zero,in_hand],[out_hand,zero,normal,zero],[zero,out_hand,zero,normal]]),index)
                    if hand=="right":
                        self.pfa.setTransmat(np.bmat([[normal,in_hand,zero,zero],[out_hand,normal,zero,zero],[zero,zero,normal,in_hand],[zero,zero,out_hand,normal]]),index)
                    if hand=="left" and move=="empty":
                        self.pfa.setTransmat(np.bmat([[normal+self.emptymatrix_diff3(cup)*difficult,zero,in_hand,zero],[zero,normal+self.emptymatrix_diff3(cup)*difficult,zero,in_hand],[out_hand,zero,normal,zero],[zero,out_hand,zero,normal]]),index)
                    if hand=="right" and move=="empty":
                        self.pfa.setTransmat(np.bmat([[normal+self.emptymatrix_diff3(cup)*difficult,in_hand,zero,zero],[out_hand,normal,zero,zero],[zero,zero,normal+self.emptymatrix_diff3(cup)*difficult,in_hand],[zero,zero,out_hand,normal]]),index)
                    index+=1
                    
        self.pfa.setTransmat(np.bmat([[zero,matrix_normal*easy,matrix_normal*difficult,zero],[zero,zero,zero,zero],[zero,zero,zero,matrix_normal*easy],[zero,zero,zero,zero]]),IN4R)
        self.pfa.setTransmat(np.bmat([[zero,matrix_normal*difficult,matrix_normal*easy,zero],[zero,zero,zero,matrix_normal*easy],[zero,zero,zero,zero],[zero,zero,zero,zero]]),IN4L)
        self.pfa.setTransmat(np.bmat([[zero,zero,zero,zero],[matrix_normal*easy,zero,zero,zero],[zero,zero,zero,zero],[zero,zero,matrix_normal*easy,zero]]),OUT4R)
        self.pfa.setTransmat(np.bmat([[zero,zero,zero,zero],[zero,zero,zero,zero],[matrix_normal*easy,zero,zero,zero],[zero,matrix_normal*easy,zero,zero]]),OUT4L)
        self.pfa.setTransmat(np.bmat([[matrix_normal*easy,zero,zero,zero],[zero,matrix_normal*easy,zero,zero],[zero,zero,zero,zero],[zero,zero,zero,zero]]),E4L)
        self.pfa.setTransmat(np.bmat([[matrix_normal*easy,zero,zero,zero],[zero,zero,zero,zero],[zero,zero,matrix_normal*easy,zero],[zero,zero,zero,zero]]),E4R)  
       # self.pfa.normalize()
                  
    def inmatrix_easy(self,cup):
        matrix=np.zeros((8,8))
        for i in range(0,8):
            for j in range(0,8):
                if(states[i][cup]=="0" and states[j][cup]=="1" and states[i][cup+1:]==states[j][cup+1:] and states[i][:cup]==states[j][:cup]):
                    matrix[i,j]=1
        return matrix
    def outmatrix_easy(self,cup):
        matrix=np.zeros((8,8))
        for i in range(0,8):
            for j in range(0,8):
                if(states[i][cup]=="1" and states[j][cup]=="0" and states[i][cup+1:]==states[j][cup+1:] and states[i][:cup]==states[j][:cup]):
                    matrix[i,j]=1
        return matrix
    def inmatrix_diff(self,cup):
        matrix=np.zeros((8,8))
        for i in range(0,8):
            for j in range(0,8):
                if(states[i][cup]=="0" and states[j]==states[i]):
                    matrix[i,j]=1
        return matrix
    def outmatrix_diff(self,cup):
        return np.identity(8)
    def emptymatrix_easy(self,cup):
        matrix=np.zeros((8,8))
        for i in range(0,8):
            for j in range(0,8):
                if(states[i][cup]=="0" and states[j]==states[i]):
                    matrix[i,j]=1
        return matrix
    def emptymatrix_diff1(self,cup):
        matrix=np.zeros((8,8))
        for i in range(0,8):
            for j in range(0,8):
                if(states[i][cup]=="1" and states[j][cup]=="0" and states[i][cup+1:]==states[j][cup+1:] and states[i][:cup]==states[j][:cup]):
                    matrix[i,j]=1
        return matrix
    def emptymatrix_diff2(self,cup):
        matrix=np.zeros((8,8))
        for i in range(0,8):
            for j in range(0,8):
                if(states[i][cup]=="0" and states[j][cup]=="1" and states[i][cup+1:]==states[j][cup+1:] and states[i][:cup]==states[j][:cup]):
                    matrix[i,j]=1
        return matrix
    def emptymatrix_diff3(self,cup):
        matrix=np.zeros((8,8))
        for i in range(0,8):
            for j in range(0,8):
                if(states[i][cup]=="1" and states[j][cup]=="1" and states[i][cup+1:]==states[j][cup+1:] and states[i][:cup]==states[j][:cup]):
                    matrix[i,j]=1
        return matrix
    def printresult(self,result):
        for r in result:
            print(self.translate(r))
    def translate(self,list):
        print(list)
        translation=[]
        for i in list:
            translation.append(states[i%len(states)]+hands[int(i/4)])    
        return translation            
                    
       

test=[E1L,E2R,E3R,E4L,E4R,E1L,IN4R,OUT1L,E1R,IN4R,OUT1L]
test=[E4L,E4R,E1R,E2L,E3R]


Logic=logic()
result=Logic.getscore(test[:1])
Logic.printresult(result[0])
p=result[2]
for i in range(1,len(test)):
    t=test[:i+1]
    result=Logic.getscore( t)
    Logic.printresult(result[0])
    p_prefix=result[2]
    p_now=result[1]
    print(p/p_prefix)
    print(p/p_now)
    p=p_now

