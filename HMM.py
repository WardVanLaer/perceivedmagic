import numpy as np
from hmmlearn import hmm





model = hmm.MultinomialHMM(n_components=36)
# predict a sequence of hidden states based on visible states
M1=np.identity(8)*0.2

M1[1,0],M1[0,1],M1[2,3],M1[3,2],M1[4,5],M1[5,4],M1[6,7],M1[7,6]=1,1,1,1,1,1,1,1
M2=np.identity(8)*0.2
M2[0,2],M2[1,3],M2[2,0],M2[3,1],M2[4,6],M2[5,7],M2[6,4],M2[7,5]=1,1,1,1,1,1,1,1
M3=np.identity(8)*0.2
M3[0,4],M3[1,5],M3[2,6],M3[3,7],M3[4,0],M3[5,1],M3[6,2],M3[7,3]=1,1,1,1,1,1,1,1

E1=M1/5+np.identity(8)*(0.96)
E1=np.concatenate((M1,E1[0::2]),0)
T1=np.concatenate((E1,E1[:,0::2]),1)
T2=np.concatenate((E1,E1[:,[0,1,4,5]]),1)
T3=np.concatenate((E1,E1[:,0:4]),1)

E2=M2/5+np.identity(8)*(0.96)
E2=np.concatenate((M2,E2[[0,1,4,5]]),0)
T4=np.concatenate((E2,E2[:,0::2]),1)
T5=np.concatenate((E2,E2[:,[0,1,4,5]]),1)
T6=np.concatenate((E2,E2[:,0:4]),1)


E3=M3/5+np.identity(8)*(0.96)
E3=np.concatenate((M3,E3[0:4]),0)
T7=np.concatenate((E3,E3[:,0::2]),1)
T8=np.concatenate((E3,E3[:,[0,1,4,5]]),1)
T9=np.concatenate((E3,E3[:,0:4]),1)

emission= np.zeros((36,9))
emission[0:8:2,0]=1
emission[1:8:2,1]=1
emission[8:12,2]=1
emission[12:20:2,3]=1
emission[13:20:2,4]=1
emission[20:24,5]=1
emission[24:32:2,6]=1
emission[25:32:2,7]=1
emission[32:,8]=1
trans=np.bmat([[T1, T2,T3], [T4, T5,T6],[T7,T8,T9]])
print(T7.shape)
rows=np.sum(trans,1)
model.transmat_=trans/rows
model.emissionprob_=emission
start=np.ones(36)/36
model.startprob_=start
#8=empty2
#7=show ball2
#6= cover ball2
#5=empty2
#4=show ball2
#3= cover ball2
#2=empty
#1=show ball
#0= cover ball

def translate(list):
    code=["000","001","010","011","100","101","110","111"]
    result=code+code[0::2]+code+code[0:2]+code[4:6]+code+code[0:4]
    translation=[]
    for i in list:
        translation.append(result[i])
    return translation

def translate2(list):
    translation=[]
    text=["IN","OUT","EMPTY"]
    for i in list:
        translation.append(text[i%3]+str(i/3))
    return translation


test=np.array([[0, 4, 5, 8, 6, 1, 2, 0, 7, 8, 1, 5, 3, 8, 0, 4, 5]]).T


for i in range(0,len(test)):
    t=test[0:i+1]
    print(translate2(t))
    print(model.predict(t))
    print(translate(model.predict(t)))
 #   print(model.score(t))
   # print(model.predict_proba(t))

print(model.transmat_[20:24,32:])
print(model.emissionprob_[30:,:])
