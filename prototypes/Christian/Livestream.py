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
kernel = np.ones((10, 25), np.uint8)

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

    while True:
        start = time.time()
        delta_x_min = 10 * delta
        delta_x_max = 300 * delta
        # read the next frame from stream
        (grabbed, frame) = vs.read()

        if not grabbed:
            print('[ERROR] unable to grab input - check connection or link')
            break
        # track_cars(frame, str(fps._numFrames))

        grayB = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if not firstImage:
            diff_image = cv2.absdiff(grayB, grayA)

            # image thresholding
            ret, thresh = cv2.threshold(diff_image, 25, 255, cv2.THRESH_BINARY)

            # image dilation
            dilated = cv2.dilate(thresh, kernel, iterations=1)
            #E = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(20,20))
            #dst = cv2.morphologyEx(thresh, cv2.MORPH_GRADIENT, E)

            # find contours
            contoursB, hierarchy = cv2.findContours(dilated.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

            # if not firstContour:
            #     for i, conB in enumerate(contoursB):
            #         for conA in contoursA:
            #             sameCon[i] = np.min(conB.mean - conA.mean)
            #
            # firstContour = False
            # contoursA = contoursB

            # add contours to original frames
            cv2.drawContours(frame, contoursB, -1, (127, 200, 0), 2)


        # cv2.putText(frame, "vehicles detected: " + str(len(cars)), (55, 115), font, 0.6, (0, 180, 0), 2)

        if not firstImage:
            cv2.imshow("Frame", cv2.resize(dilated, (800, 600)))

        #cv2.imshow("Frame", cv2.resize(frame, (800, 600)))

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
