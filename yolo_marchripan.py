############################################################################
# IMPORTS

import argparse
import os
import time

import cv2
import numpy as np
from imutils.video import FPS

import config as cfg
from centroidtracker import centroidtracker

############################################################################
# Settings
# Read settings from config.py or command line flags


#TODO ADD HELP TEXT
#TODO Standardwerte auf config sachen setzen! 
#TODO Write DB flag?
#TODO Debug Flag
cli_parser = argparse.ArgumentParser(description='Marchripan Vehicle Tracker')
cli_parser.add_argument('-headless', action = 'store_true')
cli_parser.add_argument('-write_video', action = 'store_true')
cli_parser.add_argument('-video_fps', action = 'store', type = int)
cli_parser.add_argument('-input', action = 'store', type = str)
cli_parser.add_argument('-output', action = 'store', type = str)
cli_parser.add_argument('-limit_fps', action = 'store', type = int)

args = cli_parser.parse_args()

if args.headless == False:
    HEADLESS = cfg.HEADLESS
else: 
    print('[INFO] Headless Mode - Terminate with Ctrl+C')
    HEADLESS = args.headless

if args.write_video == False:
    WRITE_VIDEO = cfg.WRITE_VIDEO
else:
    WRITE_VIDEO = args.write_video

if args.output == None: 
    OUTPUT_PATH = cfg.OUTPUT_PATH
else:
    OUTPUT_PATH = args.output

if args.limit_fps == None:
    LIMIT_FPS = cfg.LIMIT_FPS
else:
    LIMIT_FPS = args.limit_fps

if args.video_fps == None:
    if args.limit_fps == None:
        VIDEO_FPS = cfg.VIDEO_FPS
    else: VIDEO_FPS = LIMIT_FPS
else:
    VIDEO_FPS = args.video_fps

MIN_LOOP_DUR = (1/LIMIT_FPS)*1000 # In ms, to limit FPS

if WRITE_VIDEO:
    print("[INFO] Will write images into output file")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, VIDEO_FPS, (800, 600))

if args.input == None:
    YOLO_INPUT = cfg.YOLO_INPUT
else:
    YOLO_INPUT = args.input 

YOLO_PATH = cfg.YOLO_PATH

CONFIDENCE = cfg.CONFIDENCE
THRESHOLD = cfg.THRESHOLD

# load the COCO class labels our YOLO model was trained on
LABELS_PATH = cfg.LABELS_PATH
LABELS = open(LABELS_PATH).read().strip().split("\n")  

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

# derive the paths to the YOLO weights and model configuration
WEIGHTS_PATH = os.path.sep.join([YOLO_PATH, "custom-yolov4-tiny.weights"])
YOLO_CONFIG_PATH = os.path.sep.join([YOLO_PATH, "custom-yolov4-tiny.cfg"])

# Feature Toggles
REGION_OF_INTEREST = cfg.REGION_OF_INTEREST



############################################################################
# Tracking

print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(YOLO_CONFIG_PATH, WEIGHTS_PATH)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

ct = centroidtracker()


def main():
    # initialize the video stream, pointer to output video file, and
    # frame dimensions
    # initialize our centroid tracker and frame dimensions

    vs = cv2.VideoCapture(YOLO_INPUT)
    fps = FPS().start()

    (W, H) = (None, None)
    try:
        while True:
            loop_start = int(round(time.time()*1000))
            # read the next frame from stream
            (grabbed, frame) = vs.read()

            if not grabbed:
                print('[ERROR] unable to grab input - check connection or link')
                break
            track_cars(frame, "live frame: " + str(fps._numFrames))
            fps.update()
            if not HEADLESS:
                if cv2.waitKey(1) == ord('q'):
                    break
            loop_duration = loop_start = int(round(time.time()*1000)) - loop_start
            if loop_duration < MIN_LOOP_DUR: 
                time.sleep((MIN_LOOP_DUR-loop_duration)/1000)

    except KeyboardInterrupt:
        #this does not seem to work on windows but might on Ubuntu
        print('[INFO] Keyboard Interrupt - Terminating Process')
        pass

    # stop the timer and display FPS information
    fps.stop()

    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    # do a bit of cleanup
    print("[INFO] cleaning up...")
    if not HEADLESS:
        cv2.destroyAllWindows()

    # release the file pointers
    vs.release()

    if WRITE_VIDEO:
        out.release()


# process and display framesq
def track_cars(frame, frame_no):
    print("analyzing " + str(frame_no))

    # resize - skipped
    # resized_frame = cv2.resize(frame, (1280, 720))
    # height, width, channels = resized_frame.shape
    height, width, channels = frame.shape

    # process frame
    processed_frame = process_frame(frame)

    # create blob and run it through the net
    blob = cv2.dnn.blobFromImage(processed_frame, scalefactor=1 / 255.0, size=(416, 416), mean=(0, 0, 0), swapRB=True,
                                 crop=False)
    net.setInput(blob)
    results = net.forward(output_layers)
    rects = []

    # filter results and get box coordinates
    boxes, confidences, class_ids = get_boxes_from_results(results, width, height)

    # remove overlapping boxes with NMS
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE, THRESHOLD)

    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            box = np.asarray([x, y, x + w, y + h])
            rects.append(box.astype("int"))
            # draw a bounding box rectangle and label on the frame
            color = [int(c) for c in COLORS[class_ids[i]]]
            cv2.rectangle(processed_frame, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[class_ids[i]], confidences[i])
            cv2.putText(processed_frame, text, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    # update our centroid tracker using the computed set of bounding
    # box rectangles
    objects = ct.update(rects, confidences, class_ids)
    # loop over the tracked objects
    for (objectID, centroid) in objects.items():
        # draw both the ID of the object and the centroid of the
        # object on the output frame
        text = "ID {}".format(objectID)
        cv2.putText(processed_frame, text, (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(processed_frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

    # Show Frame
    if not HEADLESS:
        cv2.imshow("Frame", cv2.resize(processed_frame, (800, 600)))
    if WRITE_VIDEO: 
        out.write(cv2.resize(processed_frame, (800, 600)))



# falls es noch weiteres zum processen geben sollte.
# sonst diese methode einfach wieder löschen und direkt region of interest aufrufen
def process_frame(frame):
    if REGION_OF_INTEREST:
        frame = region_of_interest(frame)
    return frame


def region_of_interest(frame):
    mask = np.zeros_like(frame)
    vertices = np.array([[0, 195], [1280, 195], [1280, 280], [0, 280],
                         [0, 324], [1280, 324], [1280, 430], [0, 430]], np.int32)

    # fill the mask
    cv2.fillPoly(mask, [vertices], (255, 255, 255))

    # show ROI only
    masked_frame = cv2.bitwise_and(frame, mask)
    return masked_frame


def get_boxes_from_results(results, width, height):
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
    return boxes, confidences, class_ids


main()

