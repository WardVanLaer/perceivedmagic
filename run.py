import PFA
import mapper
import cv2


cap= cv2.VideoCapture('vids/test.mov')
ret,frame=cap.read()
frame_small=cv2.resize(frame,(200,200))
mapper=mapper.mapper(frame_small)
Logic=PFA.logic()
events,pre_score,pre_states=0,0,[]
while True:
    ret,frame=cap.read()
    frame_small=cv2.resize(frame,(200,200))
    frame_show=cv2.resize(frame,(700,500))
    mapper.update(frame_small)
    if len(mapper.history)>events:
        print(mapper.history)
        events=len(mapper.history)
        score=Logic.getscore(mapper.history)
        if(pre_score-score[2]>1 and events>1):
            cv2.waitKey()
        pre_score=score[2]
        print(score[1])
    frame_small=mapper.draw(frame_small)
    cv2.imshow('cups',frame_small)
    k=cv2.waitKey(1)   
    if k==13:
        break