# Test von Mean-Shift Tracking. Hier wird ein Histogramm eines erkannten Objekts erstellt welches dann in folgenden Frames gesucht wird um somit das Objekt zu tracken.
# Funktioniert für unseren Fall nicht so gut da sehr viele ähnliche Bereiche auf dem Bild zu erkennen sind. Eignet sich eher für große, sehr individuelle Objekte.


from os import walk

import cv2
import numpy as np
import os

CONFIDENCE=0.5 # probability for a certain class (std: 0.5)
THRESHOLD=0.4 # threshold used in non maximum supression (NMS) to filter out overlapping boxes (std: 0.4)

# get list of our images
IMAGE_PATH = "prototypes/meanshift/images/"
_, _, filenames = next(walk(IMAGE_PATH), (None, None, []))
print(filenames[0])

#load yolo..
print("[INFO] loading YOLO from disk...")
YOLO_PATH = "yolo-coco"
weightsPath = os.path.sep.join([YOLO_PATH, "yolov3.weights"])
configPath = os.path.sep.join([YOLO_PATH, "yolov3.cfg"])
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([YOLO_PATH, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n") #to-do only include relevant labels

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

#detect car on first image
print("read image..")
frame = cv2.imread(IMAGE_PATH+str(filenames[0]))

print("analyze image..")
height, width, channels = frame.shape 

# mask = np.zeros_like(frame)
# vertices = np.array([[0, 185], [1280, 185], [1280, 290], [0, 290],
# [0, 300], [1280, 300], [1280, 450], [0, 450]], np.int32)

# # fill the mask
# cv2.fillPoly(mask, [vertices], (255, 255, 255)) 

# # show ROI only
# frame = cv2.bitwise_and(frame, mask) 


blob = cv2.dnn.blobFromImage(frame, scalefactor=1/255.0, size=(416, 416), mean=(0, 0, 0), swapRB=True, crop=False)
net.setInput(blob)
results = net.forward(output_layers)


#get boxes from results
class_ids = [] 
confidences = []
boxes = []
# loop over each of the layer outputs
for result in results:
    # loop over each of the detections
    for detection in result:
        # extract the class ID and confidence (probability)
        # of the current object detection
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        # only take good detections 
        if confidence > CONFIDENCE:
            # Detection Coordinates
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            # Box Coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2) 
            boxes.append([x, y, w, h])
            # Save confidence and class for output 
            confidences.append(float(confidence)) 
            class_ids.append(class_id)

# remove overlapping boxes with NMS
idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE, THRESHOLD)

# ensure at least one detection exists
if len(idxs) > 0:
    # loop over the indexes we are keeping
    for i in idxs.flatten():
        # extract the bounding box coordinates
        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])

        # draw a bounding box rectangle and label on the frame
        color = [int(c) for c in COLORS[class_ids[i]]]
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        text = "{}: {:.4f} - id: {}".format(LABELS[class_ids[i]], confidences[i], i)
        cv2.putText(frame, text, (x, y - 5),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


#Show Frame
cv2.imshow("Frame", cv2.resize(frame, (800, 600)))

# while True:
#     if cv2.waitKey(1) == ord('q'):
#         break



# #now starts meanshift tracking
# cap = cv.VideoCapture()
# # take first frame of the video
# ret,frame = cap.read()

# setup initial location of window
x, y, w, h = boxes[20][0], boxes[20][1], boxes[20][2], boxes[20][3] # simply hardcoded the values
track_window = (x, y, w, h)
# set up the ROI for tracking
roi = frame[y:y+h, x:x+w]
hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
while(1):
    ret=True
    frame = cv2.imread(IMAGE_PATH+str(filenames[1])) #cap.read()
    # mask = np.zeros_like(frame)
    # vertices = np.array([[0, 185], [1280, 185], [1280, 290], [0, 290],
    # [0, 300], [1280, 300], [1280, 450], [0, 450]], np.int32)

    # # fill the mask
    # cv2.fillPoly(mask, [vertices], (255, 255, 255)) 

    # # show ROI only
    # frame = cv2.bitwise_and(frame, mask) 

    if ret == True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
        # apply meanshift to get the new location
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)
        # Draw it on image
        x,y,w,h = track_window
        img2 = cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2)
        cv2.imshow('img2',img2)
    else:
        break
    break

while(1):
    ret=True
    frame = cv2.imread(IMAGE_PATH+str(filenames[2])) #cap.read()
    # mask = np.zeros_like(frame)
    # vertices = np.array([[0, 185], [1280, 185], [1280, 290], [0, 290],
    # [0, 300], [1280, 300], [1280, 450], [0, 450]], np.int32)

    # # fill the mask
    # cv2.fillPoly(mask, [vertices], (255, 255, 255)) 

    # # show ROI only
    # frame = cv2.bitwise_and(frame, mask) 
    
    if ret == True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
        # apply meanshift to get the new location
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)
        # Draw it on image
        x,y,w,h = track_window
        img3 = cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2)
        cv2.imshow('img3',img3)
    else:
        break
    break

while(1):
    ret=True
    frame = cv2.imread(IMAGE_PATH+str(filenames[3])) #cap.read()
    # mask = np.zeros_like(frame)
    # vertices = np.array([[0, 185], [1280, 185], [1280, 290], [0, 290],
    # [0, 300], [1280, 300], [1280, 450], [0, 450]], np.int32)

    # # fill the mask
    # cv2.fillPoly(mask, [vertices], (255, 255, 255)) 

    # # show ROI only
    # frame = cv2.bitwise_and(frame, mask) 
    
    if ret == True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
        # apply meanshift to get the new location
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)
        # Draw it on image
        x,y,w,h = track_window
        img4 = cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2)
        cv2.imshow('img4',img4)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    else:
        break
    