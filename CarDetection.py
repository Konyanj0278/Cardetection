import numpy as np 
import cv2

net= cv2.dnn.readNet("yolov3.weights","yolov3.cfg")

class_ids = []
confidences = []
boxes = []
classes = []
with open("coco.txt") as f:
	classes= [line.strip() for line in f.readlines()]

layer_names= net.getLayerNames()
outputlayer = [layer_names[i[0]-1 ]for i in net.getUnconnectedOutLayers()] 
colors = np.random.uniform(0, 255, size=(len(classes), 3))
VidInput = cv2.VideoCapture(0)

while (True):
   
    ret, frame = VidInput.read()# captures individual frames for processing
    frame= cv2.resize(frame, None, fx=0.4, fy=0.4)
    height, width, channels= frame.shape
    blob= cv2.dnn.blobFromImage(frame, 0.00392, (320,320), (0,0,0), True, crop=False)
    net.setInput(blob)
    net.forward(outputlayer)
    outs = net.forward(outputlayer)
  
   
    for out in outs:
   		for detection in out:
   			scores= detection[5:]
   			class_id=np.argmax(scores)
   			confidence = scores[class_id]
   			if confidence >.5:
   				centerx= int (detection[0]* width)
   				centery= int (detection[1]* height)
   				w = int (detection[2]* width)
   				h = int (detection[3]* height)


   				x = int (centerx-w /2)
   				y = int (centery- h /2)

   				boxes.append([x,y,w,h])
   				confidences.append(float(confidence))
   				class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    print(indexes)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
        	x, y, w, h = boxes[i]
        	label = str(classes[class_ids[i]])
        	color = colors[i]
        	cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        	cv2.putText(frame, label, (x, y + 30), font, 3, color, 3)

    # Display the resulting frame
    cv2.imshow('Output',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):#Terminates loop
        break

#releases camera and closes output
cap.release()
cv2.destroyAllWindows() 