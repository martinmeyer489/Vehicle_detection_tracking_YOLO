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


RTSP_URL = URLHD

# kernel for image dilation
kernel = np.ones((8, 25), np.uint8)

# font style
font = cv2.FONT_HERSHEY_SIMPLEX

carsPerLane = [0, 0, 0, 0, 0, 0]


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

        grayB = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if not firstImage:
            diff_image = cv2.absdiff(grayB, grayA)

            # image thresholding
            ret, thresh = cv2.threshold(diff_image, 30, 255, cv2.THRESH_BINARY)

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
            # addCar(1, frame[200, 1250], rWhiteA1)
            # rWhiteA1 = np.array_equal(frame[200, 1250], np.array([255, 255, 255], None))
            # addCar(2, frame[232, 1250], rWhiteA2)
            # rWhiteA2 = np.array_equal(frame[232, 1250], np.array([255, 255, 255], None))
            # addCar(3, frame[262, 1250], rWhiteA3)
            # rWhiteA3 = np.array_equal(frame[262, 1250], np.array([255, 255, 255], None))
            #
            # subtractCar(1, frame[200, 30], lWhiteA1)
            # lWhiteA1 = np.array_equal(frame[200, 30], np.array([255, 255, 255], None))
            # subtractCar(2, frame[232, 30], lWhiteA2)
            # lWhiteA2 = np.array_equal(frame[232, 30], np.array([255, 255, 255], None))
            # subtractCar(3, frame[262, 30], lWhiteA3)
            # lWhiteA3 = np.array_equal(frame[232, 30], np.array([255, 255, 255], None))

        # for i in range(len(carsPerLane)):
        #     cv2.putText(frame, "Vehicles detected in lane " + str(i+1) + ": " + str(carsPerLane[i]), (55, 115+i*30), font, 0.6, (0, 180, 0), 2)

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


def addCar(laneNumber, pointInFrame, whiteA):
    whiteB = np.array_equal(pointInFrame, np.array([255, 255, 255], None))
    if whiteB and not whiteA:
        carsPerLane[laneNumber - 1] += 1

def subtractCar(laneNumber, pointInFrame, whiteA):
    whiteB = np.array_equal(pointInFrame, np.array([255, 255, 255], None))
    if whiteB and not whiteA:
        carsPerLane[laneNumber - 1] -= 1

main()
