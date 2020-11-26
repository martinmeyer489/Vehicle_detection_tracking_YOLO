# packages
import argparse
import datetime
import os
import time
from os import walk
from pathlib import Path

import cv2
import numpy as np
from imutils.video import FPS, VideoStream

import database as db

URL4K = "https://media.dcaiti.tu-berlin.de/tccams/1c/axis-cgi/mjpg/video.cgi"
URLHD = "https://media.dcaiti.tu-berlin.de/tccams/1c/axis-cgi/mjpg/video.cgi?camera=1&resolution=1280x720&rotation=0&audio=0&mirror=0&fps=0&compression=0"

RTSP_URL = URLHD

# kernel for image dilation
kernel = np.ones((8, 20), np.uint8)

# font style
font = cv2.FONT_HERSHEY_SIMPLEX


def main():
    # initialize the video stream, pointer to output video file, and
    # frame dimensions
    vs = cv2.VideoCapture(RTSP_URL)
    fps = FPS().start()
    writer = None
    (W, H) = (None, None)
    delta = 1
    frame_no = 0
    firstImage = True
    firstContour = True
    cars = []
    carsPerLane = [0, 0, 0, 0, 0, 0]
    rWhiteA1 = False
    rWhiteA2 = False
    rWhiteA3 = False
    lWhiteA1 = False
    lWhiteA2 = False
    lWhiteA3 = False

    while True:
        start = time.time()
        delta_x_min = 10 * delta
        delta_x_max = 300 * delta
        # read the next frame from stream
        (grabbed, frame) = vs.read()

        mask = np.zeros_like(frame)
        vertices = np.array([[0, 203], [1280, 203], [1280, 203], [0, 203],
                             [0, 235], [1280, 235], [1280, 235], [0, 235],
                             [0, 265], [1280, 265], [1280, 265], [0, 265]], np.int32)
        # fill the mask
        cv2.fillPoly(mask, [vertices], (255, 255, 255))
        # show ROI only
        masked_frame = cv2.bitwise_and(frame, mask)

        if not grabbed:
            print('[ERROR] unable to grab input - check connection or link')
            break
        # track_cars(frame, str(fps._numFrames))

        grayB = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)

        if not firstImage:
            diff_image = cv2.absdiff(grayB, grayA)

            # image thresholding
            ret, thresh = cv2.threshold(diff_image, 20, 255, cv2.THRESH_BINARY)

            # image dilation
            dilated = cv2.dilate(thresh, kernel, iterations=1)

            # form eclipse shapes
            #E = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(20,20))
            #dst = cv2.morphologyEx(thresh, cv2.MORPH_GRADIENT, E)

            # find contours
            contoursB, hierarchy = cv2.findContours(dilated.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)


            # fill contours in original frame
            cv2.fillPoly(frame,contoursB,(255,255,255))

            # add contours to original frames
            #cv2.drawContours(frame, contoursB, -1, (127, 200, 0), 2)


            # find valid contours
            rWhiteB1 = np.array_equal(frame[200, 1250], np.array([255,255,255],None))
            if rWhiteB1 and not rWhiteA1:
                carsPerLane[0] += 1
            rWhiteA1 = rWhiteB1

            rWhiteB2 = np.array_equal(frame[232, 1250], np.array([255, 255, 255], None))
            if rWhiteB2 and not rWhiteA2:
                carsPerLane[1] += 1
            rWhiteA2 = rWhiteB2

            rWhiteB3 = np.array_equal(frame[262, 1250], np.array([255, 255, 255], None))
            if rWhiteB3 and not rWhiteA3:
                carsPerLane[2] += 1
            rWhiteA3 = rWhiteB3

            
            lWhiteB1 = np.array_equal(frame[200, 20], np.array([255,255,255],None))
            if lWhiteB1 and not lWhiteA1:
                carsPerLane[0] -= 1
            lWhiteA1 = lWhiteB1

            lWhiteB2 = np.array_equal(frame[232, 20], np.array([255, 255, 255], None))
            if lWhiteB2 and not lWhiteA2:
                carsPerLane[1] -= 1
            lWhiteA2 = lWhiteB2

            lWhiteB3 = np.array_equal(frame[262, 20], np.array([255, 255, 255], None))
            if lWhiteB3 and not lWhiteA3:
                carsPerLane[2] -= 1
            lWhiteA3 = lWhiteB3



        for i in range(len(carsPerLane)):
            cv2.putText(frame, "Vehicles detected in lane " + str(i+1) + ": " + str(carsPerLane[i]), (55, 115+i*30), font, 0.6, (0, 180, 0), 2)

        if not firstImage:
             cv2.imshow("Frame", cv2.resize(dilated, (800, 600)))

        cv2.imshow("Frame", cv2.resize(frame, (800, 600)))

        grayA = grayB
        firstImage = False

        fps.update()
        if cv2.waitKey(1) == ord('q'):
            break
        now = time.time()
        delta = now - start

    # stop the timer and display FPS information
    fps.stop()

    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    # do a bit of cleanup
    print("[INFO] cleaning up...")
    cv2.destroyAllWindows()
    # release the file pointers
    vs.release()


main()
