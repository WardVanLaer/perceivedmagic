import vision_system as vision
import Logic3 as logic3

f=open("routines/man_tagged.txt","r")
numvids=int(f.readline()[:-1])
tptotal=0
selected=0
relevant=0
print("-----Creating Model-----")
surprise_detection=logic3.logic()
for i in range(numvids):
    vidname=str(f.readline()[:-1])
    print("Analyzing vid "+vidname)
    result=vision.vision_system('routines/'+vidname)
    f.readline()
    f.readline()
    truth=str(f.readline()[:-1]).split(",")
    MOI=surprise_detection.analyze(result)
    print(MOI,truth)
    tp=0
    selected+=len(MOI)
    relevant+=len(truth)
    l1=len(MOI)
    for t in truth:
        match=None
        for m in MOI:
            if abs(int(m)-int(t))<=45:
                match=m
                break
        if(match!=None):
            tp+=1
            del MOI[MOI.index(m)]
    print(result)
    print("precision= "+str(float(tp)/l1))
    print("recall="+str(float(tp)/len(truth))) 
    tptotal+=tp
print("-----Overall results-----")
print("precision= "+str(float(tptotal)/selected))
print("recall="+str(float(tptotal)/relevant))
print(relevant)


