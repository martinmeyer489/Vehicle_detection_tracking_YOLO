# Configuration file for Centroid Tracker
# All of these are loaded from the script, some of these may be overriden by flags



############################################################################
# IMPORTS
import os
import numpy as np


############################################################################
# General Settings
DEBUG_MODE = False # prints additional messages for debugging
HEADLESS = False # run without GUI on server
WRITE_VIDEO = True # write a video-file
OUTPUT_PATH = "output.avi"
VIDEO_FPS = 20  # FPS Rate of output video
LIMIT_FPS = 30  # FPS Limit for tracking-algorithm
HIDE_FRAME_COUNT = False # Dont show "Analysing Frame XYZ in console output"
                

############################################################################
# Input Settings
IS_VIDEO_INPUT = False


############################################################################
# Image Settings
REGION_OF_INTEREST = True # Skip parts of the image from being analyzed

# ROI VERTICE: Each line represents 4 points [x, y] that define a square on the input image
#              IF REGION_OF_INTEREST = TRUE, only these squares will be looked at. 
#              One can use this to exclude i.e. parking strips
#              One can add as many squares as neccesary 
#              In an 1280*720 image the top left corner would be [0, 0], the bottom right [1280, 720], ...
#              
#              One can save multiple arrays here, just make sure to edit ROI_VERTICE to load the correct 

#########
#  Centercam 1
TUB_C1 = np.array([[0, 195], [1280, 195], [1280, 280], [0, 280],
                [0, 324], [1280, 324], [1280, 430], [0, 430]], 
                np.int32)
##########
#  Centercam 2
TUB_C2 = np.array([0, 0], 
                np.int32)
##########

ROI_VERTICE = TUB_C1 # Choose the correct array from the arrays above



############################################################################
# Yolo Settings
YOLO_PATH = "yolo-coco"
LABELS_PATH = os.path.sep.join([YOLO_PATH, "coco2.names"])

WEIGHTS_PATH = os.path.sep.join([YOLO_PATH, "all_car.weights"])
YOLO_CONFIG_PATH = os.path.sep.join([YOLO_PATH, "all_car.cfg"])

CONFIDENCE = 0.01  # probability for a certain class (std: 0.5)
THRESHOLD = 0.1  # threshold used in non maximum supression (NMS) to filter out overlapping boxes (std: 0.4)


############################################################################
# Tracker Settings
DISTANCE_TOLERANCE = 140 # distance tolerance for assign IDs (Object centroid to Input centroid)
VERTICAL_TOLERANCE = 20 # vertical tolerance for assign IDs (Object centroid to Input centroid)

MAX_DISAPPEARED = 20 # Number of consecutive frames without detection before deregistering an object

# The next three configs are very specific to the cameras used for this project. They might need to be heavily
# adjusted for different camera setups or the corresponding code in the centroidtracker.py might need to be 
# changed completely for different camera setups etc.

IGNORE_REGISTRATION_ZONES = False # Setting this to true will skip the check whether the vehicle is in a valid position for (de-)registration
                                  # This is useful when testing other camera inputs where registration zones are not applyable or not yet configured

LANE_SEPARATOR = 300 # <300 are the upper lanes (eastbound), >300 the lower (westbound)

DEREGISTRATION_ZONE = 50 # If the centroid of a car is within 50px of frame-edge, it will be deregistered

FRAME_WIDTH = 1280 # Needed to calculate the deregistration zone for the lower lanes



############################################################################
# Database Settings
SKIP_DB = False
DATABASE_PATH = r"database.db"

CLASSES = [ # List of Lists [class_id, class_name], used to initialize classes DB. Should be like coco.names
                [0, "car"]
            ]


