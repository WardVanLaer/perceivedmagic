import vision_system as vision


f=open("tests/tagged.txt","r")
numvids=int(f.readline()[:-1])
tptotal=0
selected=0
relevant=0
for i in range(numvids):
    
    vidname=str(f.readline()[:-1])
    print("Analyzing vid "+vidname)
    events=f.readline()[:-1].split(",")
    times=f.readline()[:-1].split(",")
    result=vision.vision_system('tests/'+vidname)
    tp=0
    selected+=len(result[0])
    relevant+=len(events)
    for i in range(len(events)):
        for j in range(len(result[0])):
            if abs(result[1][j]-int(times[i]))<=60 and events[i]==result[0][j]:
                tp+=1
                break
    
    print(result)
    print("precision= "+str(float(tp)/len(result[0])))
    print("recall="+str(float(tp)/len(events))) 
    tptotal+=tp
print("-----Overall results-----")
print("precision= "+str(float(tptotal)/selected))
print("recall="+str(float(tptotal)/relevant))
print(relevant)

