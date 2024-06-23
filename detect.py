import cv2
import numpy as np
import urllib.request

url = 'http://192.168.43.219/cam-hi.jpg'

cap = cv2.VideoCapture(url)
whT=320
confThreshold = 0.5
nmsThreshold = 0.3
classesfile='custom.names'
classNames=[]
with open(classesfile,'rt') as f:
    classNames=f.read().rstrip('\n').split('\n')
#print(classNames)

modelConfig = 'yolov5x.cfg'
modelWeights= 'yolov5x.weights'
net = cv2.dnn.readNetFromDarknet(modelConfig,modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
def findObject(outputs,im):
    hT,wT,cT = im.shape
    bbox = []
    classIds = []
    confs = []
    found_grass = False
    found_bitter_gourd = False
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w,h = int(det[2]*wT), int(det[3]*hT)
                x,y = int((det[0]*wT)-w/2), int((det[1]*hT)-h/2)
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))
    #print(len(bbox))
    indices = cv2.dnn.NMSBoxes(bbox,confs,confThreshold,nmsThreshold)
    print(indices)
   
    for i in indices:
        i = i[0]
        box = bbox[i]
        x,y,w,h = box[0],box[1],box[2],box[3]
        if classNames[classIds[i]] == 'grass':
            found_grass = True
        elif classNames[classIds[i]] == 'bitter_gourd':
            found_bitter_gourd = True
            
        if classNames[classIds[i]]=='grass':
            
            cv2.rectangle(im,(x,y),(x+w,y+h),(255,0,255),2)
            cv2.putText(im, f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%', (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,255), 2)
            print('grass')
            print(found_grass)
            
        if classNames[classIds[i]]=='bitter_gourd':
             
            cv2.rectangle(im,(x,y),(x+w,y+h),(255,0,255),2)
            cv2.putText(im, f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%', (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,255), 2)
            print('bitter_gourd')
            print(found_bitter_gourd)
            
            
        if found_grass and found_bitter_gourd:
            print('alert')


while True:
    img_resp=urllib.request.urlopen(url)
    imgnp=np.array(bytearray(img_resp.read()),dtype=np.uint8)
    im = cv2.imdecode(imgnp,-1)
    sucess, img= cap.read()
    blob=cv2.dnn.blobFromImage(im,1/255,(whT,whT),[0,0,0],1,crop=False)
    net.setInput(blob)
    layernames=net.getLayerNames()
    #print(layernames)
    outputNames = [layernames[i[0]-1] for i in net.getUnconnectedOutLayers()]

    #print(net.getUnconnectedOutLayers())
    outputs = net.forward(outputNames)
    #print(outputs[0].shape)
    #print(outputs[1].shape)
    #print(outputs[2].shape)
    #print(outputs[0][0])
    findObject(outputs,im)



    cv2.imshow('IMage',im)
    cv2.waitKey(1)