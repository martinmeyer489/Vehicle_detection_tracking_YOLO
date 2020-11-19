#packages
import numpy as np
import cv2
#import database_connection as db 
import datetime
import argparse
import os
import time
from os import walk
from pathlib import Path




# reads classes, which should be tracked from classes.txt
def get_classes_from_config(class_config):
    global f
    with open(class_config, "r") as f:
        return [line.strip() for line in f.readlines()]
# analyze image using yolo
def analyze_image(image):
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    return net.forward(output_layers)
# gets relevant classes from all classes
def get_relevant_class_ids():
    class_ids = []
    for relevant_class in relevant_classes:
        class_ids.append(classes.index(relevant_class))
    return class_ids
# remove overlapping boxes
def remove_overlapping_boxes(boxes, confidences):
    if len(boxes) > 0:
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        for i in range(len(boxes)):
            if i not in indices:
                del boxes[i]
    return boxes



# define region of interest and apply as a mask on image 
def define_roi(image):
    # blank mask
    mask = np.zeros_like(image)
    vertices = np.array([[0, 185], [1280, 185], [1280, 290], [0, 290],
    [0, 300], [1280, 300], [1280, 450], [0, 450]], np.int32)
    # fill the mask
    cv2.fillPoly(mask, [vertices], (255, 255, 255)) 
    # show ROI only
    masked_img = cv2.bitwise_and(image, mask) 
    return masked_img


#use results delivered by yolo to build bounding boxes
def get_boxes_from_results(results, relevant_class_ids, width, height):
    class_ids = [] 
    confidences = []
    boxes = []
    for result in results:
        for detection in result:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            # only take good detections 
            if confidence > 0.5:
                # filter only detect cars, trucks, motorbikes 
                if class_id in relevant_class_ids:
                    # Detection Coordinates Hier könnte man die nehmen für DB 
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


# show boxes on top of image
def show_boxes(image, boxes, class_ids, vehicle_ids):
    # show boxes and labels 
    for i in range(len(boxes)):
        x, y, w, h = boxes[i]
        # draw a bounding box rectangle and label on the frame in rnd color each class 
        cv2.rectangle(image, (x, y), (x + w, y + h), 0, 2)
        text = "{} {}: {:.1f}kmh".format(classes[class_ids[i]], vehicle_ids[i][0], vehicle_ids[i][1]) 
        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 2)


# write results in database
def persist_results(boxes, class_ids, confidences, image):
    now = datetime.datetime.now().isoformat()
    frame_id = db.insert_frame(db_connection, (image, delta, now)) 
    # show boxes and labels
    vehicles = []
    for i in range(len(boxes)):
        x, y, w, h = boxes[i]
        class_name = str(classes[class_ids[i]])
        confidence = str(round(confidences[i], 2))
        # if already found use same vehicle_id
        vehicle = check_if_vehicle_already_found(frame_id, class_name, x, y, w, h, ) 
        db.insert_position(db_connection, (vehicle[0], frame_id, x, y, w, h, confidence, now)) 
        vehicles.append(vehicle)
    return vehicles


# checks for the given coordinates, whether it could be one of the previous found vehicles 
def check_if_vehicle_already_found(frame_id, class_name, x, y, w, h):
    now = datetime.datetime.now().isoformat() 
    if frame_id > 1:
        previous_frame = db.select_previous_frame(db_connection, frame_id) 
        previous_frame_id = previous_frame[0]
        positions = db.select_all_positions_by_frame_id(db_connection, previous_frame_id) 
        for position in positions:
            pos_vehicle_id = position[1] 
            pos_x = position[3]
            pos_y = position[4]
            pos_w = position[5]
            pos_h = position[6]
            # compare width and height is within delta (not used as the size is quite different sometimes)
            # if abs(int(pos_w) - int(w)) < int(delta_w) and abs(int(pos_h) - int(h)) < int(delta_h): 
            # check y position is within delta
            if abs(int(pos_y) - int(y)) < int(delta_y):
            # if in top side check for negative delta 
                if int(pos_y) > 300:
                    if -int(delta_x_max) < (int(pos_x) - int(x)) < -int(delta_x_min):
                        speed = calculate_speed(pos_x, x, pos_vehicle_id)
                        return pos_vehicle_id, speed
                else:
                    if int(delta_x_min) < (int(pos_x) - int(x)) < int(delta_x_max):
                        speed = calculate_speed(pos_x, x, pos_vehicle_id)
                        return pos_vehicle_id, speed
    # if not found check in second previous frame
    if frame_id > 2:
        second_prev_frame = db.select_previous_frame(db_connection, previous_frame_id)
        second_prev_frame_id = second_prev_frame[0]
        prev_positions = db.select_all_positions_by_frame_id(db_connection,second_prev_frame_id)
        for position in prev_positions:
            pos_vehicle_id = position[1]
            pos_x = position[3]
            pos_y = position[4]
            # check y position is within delta
            if abs(int(pos_y) - int(y)) < int(delta_y):
                # if in top side check for negative delta 
                if int(pos_y) > 300:
                    if -2 * int(delta_x_max) < (int(pos_x) - int(x)) < -2*int(delta_x_min):
                        speed = calculate_speed(pos_x, x, pos_vehicle_id)
                        return pos_vehicle_id, speed
                else:
                    if 2 * int(delta_x_min) < (int(pos_x) - int(x)) < 2 * int(delta_x_max):
                        speed = calculate_speed(pos_x, x, pos_vehicle_id)
                        return pos_vehicle_id, speed
    return db.insert_vehicle(db_connection, (class_name, now)), 0


# calculates Speed from pixels and delta(input) 
def calculate_speed(x_start, x_end, vehicle_id):
    # calculate current speed
    speed = (abs(int(x_start) - int(x_end)) / 16) / float(delta) 
    speed_kmh = speed * 3.6
    # calculate average speed of vehicle
    prev_speed = db.select_average_speed_by_vehicle_id(db_connection, vehicle_id) 
    frame_number = db.count_positions_by_vehicle_id(db_connection, vehicle_id) 
    avg_speed = speed_kmh
    if prev_speed is not None:
        avg_speed = (prev_speed * speed_kmh) / (frame_number + 1)
    # persist in db
    db.update_average_speed_by_vehicle_id(db_connection, vehicle_id, avg_speed) # return current speed for output
    return speed_kmh


# tracks cars in images
def track_cars(image, image_path):
    print("analyze " + str(image_path))
    # resize
    resized_image = cv2.resize(image, (1280, 720)) 
    height, width, channels = resized_image.shape 
    # use roi
    img_roi = define_roi(resized_image)
    # Yolo magic
    results = analyze_image(img_roi)
    # get relevant classes
    relevant_class_ids = get_relevant_class_ids() 
    # display boxes
    boxes, confidences, class_ids = get_boxes_from_results(results, relevant_class_ids, width, height) 
    # remove overlapping boxes
    boxes = remove_overlapping_boxes(boxes, confidences)
    # write results to DB
    vehicle_ids = persist_results(boxes, class_ids, confidences, str(image_path)) 
    # display boxes
    show_boxes(resized_image, boxes, class_ids, vehicle_ids) 
    cv2.imshow(str(image_path), resized_image)




# construct the argument parse and parse the arguments 
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=False,
                help="path to input (folder if pictures)")
ap.add_argument("-t", "--type", required=True,
                help="type of input (live, image, video")
ap.add_argument("-d", "--delta", required=False,
                help="time between the pictures in seconds")
args = vars(ap.parse_args())
# Load Yolo
net = cv2.dnn.readNet("yolo-coco/yolov3.weights", "yolo-coco/yolov3.cfg") 
relevant_classes = ["car", "truck", "bus", "motorbike"]
# Get classes and layers
classes = get_classes_from_config("classes.txt")
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
delta_y = 10
# Setup DB
db_connection = db.setup()
input_type = args["type"] 
print("Input Type: " + input_type)



if input_type == "image":
    delta = args["delta"]
    # thresholds how much a vehicle can differ from prev frame
    delta_x_min = 100 * float(delta)
    delta_x_max = 300 * float(delta)
    # Get paths from argument
    path = Path(__file__).parent / args["input"]
    print("Reading from " + str(path))
    f = []
    _, _, filenames = next(walk(path), (None, None, []))
    # Load images and start tracking
    for filename in sorted(filenames):
        img = cv2.imread(str(path / filename))
        track_cars(img, path / filename)
        if cv2.waitKey(10) == ord('q'):
            break


if input_type == "video":
    video_path = args["input"]
    vs = cv2.VideoCapture(video_path)
    frame_no = 0
    prev_frame_no = 0
    delta = 1
    while True:
        # read the next frame from the file
        (grabbed, frame) = vs.read()
        if not grabbed:
            break
        if frame_no > 0:
            frame_diff = cv2.absdiff(prev_frame, frame)
            diff = np.mean(frame_diff)
            if diff > 1:
                delta = (frame_no - prev_frame_no) / 60
                print(delta)
                delta_x_min = 10 * delta
                delta_x_max = 300 * delta
                track_cars(frame, video_path + " frame:" + str(frame_no))
                prev_frame_no = frame_no
            else:
                track_cars(frame, video_path + " frame:" + str(frame_no))
            prev_frame = frame
            frame_no = frame_no + 1
            if cv2.waitKey(10) == ord('q'):
                break

if input_type == "stream":
    vs = cv2.VideoCapture("https://media.dcaiti.tu-berlin.de/tccams/1c/axis-cgi/mjpg/video.cgi?camera=1&resolution=1280x720&rotation=0&audio=0&mirror=0&fps=0&compression=0")
    delta = 1
    frame_no = 0
    while True:
        start = time.time()
        delta_x_min = 10*delta
        delta_x_max = 300*delta
        # read the next frame from stream
        (grabbed, frame) = vs.read()
        if not grabbed:
            break
        track_cars(frame, "live frame:" + str(frame_no))
        frame_no = frame_no + 1
        if cv2.waitKey(1) == ord('q'):
            break
        now = time.time()
        delta = now-start