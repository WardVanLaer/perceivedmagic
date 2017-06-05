

import numpy as np



class model:
    def __init__(self,states,inputs):
        self.transmats=[np.identity(states) for i in range(0,inputs)]
        self.states=states
        self.startProb=np.ones(self.states)/states
        return
    def setTransmat(self,mat,input):
        self.transmats[input]=mat
    def normalize(self):
        for s in range(self.states):
            sum=0
            for mat in self.transmats:
                sum+=np.sum(mat[s,:])
            print(sum/1)
            for mat in self.transmats:
                mat[s,:]=mat[s,:]/sum
        
        return True
    def viterbi_decode(self,input):
        t=len(input)
        s=self.states
        V=np.zeros((t, s))
        correction=1
        # store back-references to figure out the actual path taken to get there
        P=[[[] for j in range (s)] for i in range(t)]
      #  P = np.empty((t,s))
         # iterate through each time step
        for i in range(0,t):
            correction+=0
            # and consider each state
            for state in range(0,s):
            # get array of  probabilities of having been in each previous step,
            # multiplied by the probability of transitioning to the current state
                X = np.zeros(s)
                for x in range(0,s):
                    A=self.transmats[input[i]]
                    if i==0:
                        X[x] = self.startProb[x]*A[x,state]
                    else:   
                        X[x] = V[i-1, x]*A[x,state]*correction
                V[i, state] = np.max(X)
                P[i][state] = list(np.argwhere(X == np.amax(X)).flatten())
        start=list(np.argwhere(V[i] == np.amax(V[i])).flatten())
        result=[]
        for s in start:
            stop=False
            while stop==False:
                states=[int(s)]
                stop=True
                for i in range(t-1,-1,-1):
                    poss=P[i][states[0]]     
                    states=[int(poss[0])]+states
                    if(len(poss)>1):
                        stop=False
                        del poss[0]
                result.append(states)       
       # print(V)
        s=V[len(V)-2]
        p_prefix=[]
        for r in result:
            p_prefix.append(s[r[len(r)-2]])
        return (result,np.max(V[-1]),np.min(p_prefix))
    
    
    
    
    
