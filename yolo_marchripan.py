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

#TODO Continue Movement bei ObjectID 29 genauer anschauen vom Video
#TODO Parameter abchecken

cli_parser = argparse.ArgumentParser(description='''Marchripan Vehicle Tracker - 
                                                All arguments can be set permanently in config.py and need only to  
                                                be used for quick changes or debugging purposes.
                                                You can abbreviate the arguments as long as they stay distinct 
                                                (i.e. -write instead of -write_video''')
cli_parser.add_argument('-headless', action = 'store_true', default = cfg.HEADLESS, 
                        help = 'Do not show output image on screen (for server)')
cli_parser.add_argument('-write_video', action = 'store_true', default = cfg.WRITE_VIDEO, 
                        help = 'Write output to a video file')
cli_parser.add_argument('-video_fps', action = 'store', type = int, default = cfg.VIDEO_FPS,
                        help = 'FPS of output video file')
cli_parser.add_argument('-limit_fps', action = 'store', type = int, default = cfg.LIMIT_FPS,
                        help = 'Limit Number of Frames tracked per second')
cli_parser.add_argument('-input', action = 'store', type = str, default = cfg.YOLO_INPUT,
                        help = 'Input (URL or File Path) if different from config.py')
cli_parser.add_argument('-output', action = 'store', type = str, default = cfg.OUTPUT_PATH,
                        help = 'Output file path if different from config.py')
cli_parser.add_argument('-skip_db', action = 'store_true', default = cfg.SKIP_DB,
                        help = 'Set if writing to DB should be deactivated')
cli_parser.add_argument('-debug_output', action = 'store_true', default = cfg.DEBUG_MODE,
                        help = 'Set if additional Debug output should be printed')
cli_parser.add_argument('-hide_frame_count', action = 'store_true', default = cfg.HIDE_FRAME_COUNT,
                        help = 'Set if FPS count should be omitted. Useful for debugging.')
cli_parser.add_argument('-roi_off', action = 'store_false', default = cfg.REGION_OF_INTEREST,
                        help = 'Turns off region_of_interest. The whole frame will be analyzed.')
cli_parser.add_argument('-ignore_registration_zones', action = 'store_true', default = cfg.IGNORE_REGISTRATION_ZONES,
                        help = 'Stop checking whether the car is in a valid position to be (de-)registrated.')

args = cli_parser.parse_args()

cfg.HEADLESS = args.headless
cfg.WRITE_VIDEO = args.write_video
cfg.OUTPUT_PATH = args.output
cfg.LIMIT_FPS = args.limit_fps
cfg.YOLO_INPUT = args.input
cfg.SKIP_DB = args.skip_db
cfg.DEBUG_MODE = args.debug_output
cfg.HIDE_FRAME_COUNT = args.hide_frame_count
cfg.REGION_OF_INTEREST = args.roi_off
cfg.IGNORE_REGISTRATION_ZONES = args.ignore_registration_zones

if args.video_fps != None:
    cfg.VIDEO_FPS = args.video_fps
    if cfg.VIDEO_FPS != cfg.LIMIT_FPS:
        print("[INFO] Writing and tracking FPS are different!")
else:
    cfg.VIDEO_FPS = cfg.LIMIT_FPS

if cfg.HEADLESS:
    print('[INFO] Headless Mode - Terminate with Ctrl+C')

if cfg.WRITE_VIDEO:
    print("[INFO] Will write images into output file")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(cfg.OUTPUT_PATH, fourcc, cfg.VIDEO_FPS, (800, 600))

if cfg.HIDE_FRAME_COUNT:
    print('[INFO] FPS Count hidden - will not output current frame!')

if not cfg.YOLO_INPUT.startswith("http"):
    cfg.IS_VIDEO_INPUT = True
    print('[INFO] Video Input detected. Some functions (continue_movement) may behave differently')

if not cfg.REGION_OF_INTEREST:
    print('[INFO] REGION OF INTEREST is currently deactivated')

if cfg.IGNORE_REGISTRATION_ZONES:
    print('[INFO] Registration zones are currently deactivated')

MIN_LOOP_DUR = (1/cfg.LIMIT_FPS)*1000 # In ms, to limit FPS

YOLO_PATH = cfg.YOLO_PATH

CONFIDENCE = cfg.CONFIDENCE
THRESHOLD = cfg.THRESHOLD

# load the COCO class labels our YOLO model was trained on
LABELS_PATH = cfg.LABELS_PATH
LABELS = open(LABELS_PATH).read().strip().split("\n")  

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")


############################################################################
# Tracking

print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(cfg.YOLO_CONFIG_PATH, cfg.WEIGHTS_PATH)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

ct = centroidtracker()


def main():
    # initialize the video stream, pointer to output video file, and
    # frame dimensions
    # initialize our centroid tracker and frame dimensions

    vs = cv2.VideoCapture(cfg.YOLO_INPUT)
    fps = FPS().start()

    try:
        while True:
            loop_start = int(round(time.time()*1000))
            # read the next frame from stream
            (grabbed, frame) = vs.read()

            if not grabbed:
                print('[ERROR] unable to grab input - check connection or link')
                break
            track_cars(frame, "frame: " + str(fps._numFrames))
            fps.update()
            if not cfg.HEADLESS:
                if cv2.waitKey(1) == ord('q'):
                    print('[INFO] Keyboard Interrupt - Terminating Process')
                    break
            loop_duration = loop_start = int(round(time.time()*1000)) - loop_start
            if loop_duration < MIN_LOOP_DUR: 
                time.sleep((MIN_LOOP_DUR-loop_duration)/1000)

    except KeyboardInterrupt:
        #this will not work correctly on windows but does on Ubuntu
        print('[INFO] Keyboard Interrupt - Terminating Process')
        pass

    # stop the timer and display FPS information
    fps.stop()

    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    # do a bit of cleanup
    print("[INFO] cleaning up...")
    if not cfg.HEADLESS:
        cv2.destroyAllWindows()

    # release the file pointers
    vs.release()

    if cfg.WRITE_VIDEO:
        out.release()


# process and display framesq
def track_cars(frame, frame_no):
    if not cfg.HIDE_FRAME_COUNT:
        print("[INFO] Analyzing " + str(frame_no))

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
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[class_ids[i]], confidences[i])
            cv2.putText(frame, text, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.putText(frame, str(frame_no), (10, 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    # update our centroid tracker using the computed set of bounding
    # box rectangles
    objects = ct.update(rects, confidences, class_ids)
    # loop over the tracked objects
    for (objectID, centroid) in objects.items():
        # draw both the ID of the object and the centroid of the
        # object on the output frame
        text = "ID {}".format(objectID)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

    # Show Frame
    if not cfg.HEADLESS:
        cv2.imshow("Frame", cv2.resize(frame, (800, 600)))
    if cfg.WRITE_VIDEO: 
        out.write(cv2.resize(frame, (800, 600)))



# falls es noch weiteres zum processen geben sollte.
# sonst diese methode einfach wieder löschen und direkt region of interest aufrufen
def process_frame(frame):
    if cfg.REGION_OF_INTEREST:
        frame = region_of_interest(frame)
    return frame


def region_of_interest(frame):
    mask = np.zeros_like(frame)

    # fill the mask
    cv2.fillPoly(mask, [cfg.ROI_VERTICE], (255, 255, 255))

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

