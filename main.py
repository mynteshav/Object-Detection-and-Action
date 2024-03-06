import cv2
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import datetime
config_file="ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
frozen_model="frozen_inference_graph.pb"
model=cv2.dnn_DetectionModel(frozen_model,config_file)
classLables=[]
file_name='labels.txt'
with open(file_name,'rt') as fpt:
    classLabels=fpt.read().rstrip('\n').split('\n')
model.setInputSize(320,320)
model.setInputScale(1.0/320)
model.setInputMean((320,320,320))
model.setInputSwapRB(True)
cap=cv2.VideoCapture('rtsp://admin:Reset123$@192.168.1.64/1')
if not cap.isOpened():
    cap=cv2.VideoCapture('rtsp://admin:Reset123$@192.168.1.64/1')
if not cap.isOpened():
    raise IOError('cant open the webcam')
font_scale=3
font=cv2.FONT_HERSHEY_PLAIN
while True:
    ret,frame=cap.read()
    if not ret:
        print("Error: Frame not received from the video source.")
        continue
    frame=cv2.resize(frame,(320,320))
    ClassIndex,confidece,bbox=model.detect(frame,confThreshold=0.55)
    print(ClassIndex)
    if(len(ClassIndex)!=0):
        for ClassInd,conf,boxes in zip(ClassIndex.flatten(),confidece.flatten(),bbox):
            if(ClassInd<=80):
                cv2.rectangle(frame,boxes,(255,0,0),2)
                cv2.putText(frame,classLabels[ClassInd-1],(boxes[0]+10,boxes[1]+40),font,fontScale=font_scale,color=(0,255,0),thickness=3)
                if classLabels[ClassInd - 1] == 'person':
                    print("Welcome")
    cv2.imshow('object Detection',frame)
    
    if cv2.waitKey(1) & 0xff == ord('q'):
        break 

cap.release()
cv2.destroyAllWindows()