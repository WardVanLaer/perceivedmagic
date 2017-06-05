import PFA
import numpy as np

IN1,OUT1,E1,IN2,OUT2,E2,IN3,OUT3,E3=range(0,9)
states=["000", "100", "010", "110", "001", "101", "011", "111"]

class logic:
    def __init__(self):
        self.pfa=PFA.model(8,9)
        self.fill()
        
    def getscore(self,input):
        return self.pfa.viterbi_decode(input)
    def fill(self):
        a=0.12
        b=0.03
        for i in [IN1,IN2,IN3]:
            self.pfa.setTransmat(self.inmatrix_easy(i/3)*a+self.inmatrix_diff(i/3)*b,i)
        for i in [OUT1,OUT2,OUT3]:
            self.pfa.setTransmat(self.outmatrix_easy(i/3)*a+self.outmatrix_diff(i/3)*b,i)
        for i in [E1,E2,E3]:
            self.pfa.setTransmat(self.emptymatrix_easy(i/3)*a+self.emptymatrix_diff(i/3)*b,i)
       # self.pfa.normalize()
        return
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
    def emptymatrix_diff(self,cup):
        matrix=np.zeros((8,8))
        for i in range(0,8):
            for j in range(0,8):
                if(states[i][cup]=="0" and states[j][cup]=="1" and states[i][cup+1:]==states[j][cup+1:] and states[i][:cup]==states[j][:cup]):
                    matrix[i,j]=1
                if(states[i][cup]=="1" and states[j][cup]=="1" and states[i][cup+1:]==states[j][cup+1:] and states[i][:cup]==states[j][:cup]):
                    matrix[i,j]=1
                if(states[i][cup]=="1" and states[j][cup]=="0" and states[i][cup+1:]==states[j][cup+1:] and states[i][:cup]==states[j][:cup]):
                    matrix[i,j]=1
        return matrix
        
        
        
    


test=[E1,E2,E3,OUT1,OUT2,OUT3]

Logic=logic()
result=Logic.getscore(test[:1])
p=result[2]
for i in range(1,len(test)):
    t=test[:i+1]
    result=Logic.getscore( t)
    p_prefix=result[3]
    p_now=result[2]
    print(p/p_prefix)
    print(p/p_now)
    p=p_now
    
