# Configuration file for Centroid Tracker
# All of these are loaded from the script, some of these may be overriden by flags



############################################################################
# IMPORTS
import os


############################################################################
# General Settings
DEBUG_MODE = False # prints additional messages for debugging
HEADLESS = False # run without GUI on server
WRITE_VIDEO = False # write a video-file
OUTPUT_PATH = "output.avi"
VIDEO_FPS = 10 #chose a value roughly matching stream fps 
                # if input is video then set to video fps

############################################################################
# Input Settings
YOLO_INPUT = "https://media.dcaiti.tu-berlin.de/tccams/1c/axis-cgi/mjpg/video.cgi?camera=1&resolution=1280x720&rotation=0&audio=0&mirror=0&fps=0&compression=00"


############################################################################
# Image Settings
REGION_OF_INTEREST = True


############################################################################
# Yolo Settings
YOLO_PATH = "yolo-coco"
LABELS_PATH = os.path.sep.join([YOLO_PATH, "coco2.names"])

WEIGHTS_PATH = os.path.sep.join([YOLO_PATH, "custom-yolov4-tiny.weights"])
YOLO_CONFIG_PATH = os.path.sep.join([YOLO_PATH, "custom-yolov4-tiny.cfg"])

CONFIDENCE = 0.01  # probability for a certain class (std: 0.5)
THRESHOLD = 0.1  # threshold used in non maximum supression (NMS) to filter out overlapping boxes (std: 0.4)


############################################################################
# Tracker Settings
VERTICAL_TOLERANCE_MOVEMENT = 5 # vertical tolerance for continuing movement of unassgined IDs (previous to preprevious)
VERTICAL_TOLERANCE = 10 # vertical tolerance for assign IDs (Object centroid to Input centroid)
MAX_DISAPPEARED = 5 # Number of consecutive frames without detection before deregistering an object


############################################################################
# Database Settings
DATABASE_PATH = r"database.db"


