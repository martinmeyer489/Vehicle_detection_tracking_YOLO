from collections import OrderedDict
import numpy as np
from scipy.spatial import distance as dist

# vertical tolerance for continuing movement of unassgined IDs (previous to preprevious)
verticalToleranceMovement = 5

# vertical tolerance for assign IDs (Object centroid to Input centroid)
verticalTolerance = 10

maxDisappeared = 5

class centroidtracker():
    def __init__(self, maxDisappeared = maxDisappeared):
        # initialize the next unique object ID along with two ordered
        # dictionaries used to keep track of mapping a given object
        # ID to its centroid and number of consecutive frames it has
        # been marked as "disappeared", respectively
        self.previousPos = OrderedDict()
        self.pre_previousPos = OrderedDict()
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.missing_detection = OrderedDict()

        # store the number of maximum consecutive frames a given
        # object is allowed to be marked as "disappeared" until we
        # need to deregister the object from tracking
        self.maxDisappeared = maxDisappeared

    def register(self, centroid):
        # when registering an object we use the next available object
        # ID to store the centroid
        self.objects[self.nextObjectID] = centroid
        self.previousPos[self.nextObjectID] = None
        self.pre_previousPos[self.nextObjectID] = None
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1
        self.missing_detection[self.nextObjectID] = False

    def deregister(self, objectID):
        # to deregister an object ID we delete the object ID from
        # both of our respective dictionaries
        del self.objects[objectID]
        del self.disappeared[objectID]
        del self.previousPos[objectID]
        del self.pre_previousPos[objectID]

    def update(self, rects):
        #create current timestamp
        #.......

        # check to see if the list of input bounding box rectangles
        # is empty
        if len(rects) == 0:
            # loop over any existing tracked objects and mark them
            # as disappeared
            for objectID in list(self.disappeared.keys()):
                self.missing_detection[self.nextObjectID] = False
                self.disappeared[objectID] += 1
                self.continueMovement(objectID, verticalToleranceMovement)
                self.pre_previousPos[objectID] = self.previousPos[objectID]
                self.previousPos[objectID] = self.objects[objectID]
                # if we have reached a maximum number of consecutive
                # frames where a given object has been marked as
                # missing, deregister it
                # first two ifs check if vehicle is on edge of lane and use different maxdisappeared if so
                if self.objects[objectID][1] < 277 and self.objects[objectID][0] < 50:
                    self.deregister(objectID)
                elif self.objects[objectID][1] > 327 and self.objects[objectID][0] > 1230:
                    self.deregister(objectID)
                else:
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)


            # return early as there are no centroids or tracking info
            # to update
            
            #pass to insert_funktion:
            #object_id -> key von self.objects, und die jeweiligen values des keys ist ein array mit [x, y]
            #bounding boxes: rects hat die 4 koordinaten von der bb, siehe yolo_marchripan zeile 139
            #object_type: siehe zeile 144 von yolo_marchripan
            # confidence: siehe selbe zeile
            # 

            return self.objects

        # initialize an array of input centroids for the current frame
        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        # loop over the bounding box rectangles
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            # use the bounding box coordinates to derive the centroid
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)
        # if we are currently not tracking any objects take the input
        # centroids and register each of them
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                # check if vehicle is not on edge of lane
                if inputCentroids[i][1] < 277 and inputCentroids[i][0] >= 50:
                    self.register(inputCentroids[i])
                if inputCentroids[i][1] > 327 and inputCentroids[i][0] <= 1230:
                    self.register(inputCentroids[i])
                # otherwise, are are currently tracking objects so we need to
                # try to match the input centroids to existing object centroids
        else:
            # grab the set of object IDs and corresponding centroids
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            # compute the distance between each pair of object
            # centroids and input centroids, respectively -- our
            # goal will be to match an input centroid to an existing
            # object centroid
            D = dist.cdist(np.array(objectCentroids), inputCentroids)
            # in order to perform this matching we must (1) find the
            # smallest value in each row and then (2) sort the row
            # indexes based on their minimum values so that the row
            # with the smallest value is at the *front* of the index
            # list
            rows = D.min(axis=1).argsort()
            # next, we perform a similar process on the columns by
            # finding the smallest value in each column and then
            # sorting using the previously computed row index list
            cols = D.argmin(axis=1)[rows]
            # in order to determine if we need to update, register,
            # or deregister an object we need to keep track of which
            # of the rows and column indexes we have already examined
            usedRows = set()
            usedCols = set()
            # loop over the combination of the (row, column) index
            # tuples
            for (row, col) in zip(rows, cols):
                # if we have already examined either the row or
                # column value before, ignore it
                # val
                if row in usedRows or col in usedCols:
                    continue
                # otherwise, grab the object ID for the current row,
                # set its new centroid, and reset the disappeared
                # counter

                #check if distance between objectCentroid and detetcted car is smaller than certain value
                if D[row][col] < 200:
                    #check if vertical distance is smaller than certain value
                    if abs(self.objects[objectIDs[row]][1] - inputCentroids[col][1]) < verticalTolerance:
                        objectID = objectIDs[row]
                        self.objects[objectID] = inputCentroids[col]
                        self.disappeared[objectID] = 0
                        self.missing_detection[objectID] = False

                        self.pre_previousPos[objectID] = self.previousPos[objectID]
                        self.previousPos[objectID] = self.objects[objectID]
                        # indicate that we have examined each of the row and
                        # column indexes, respectively
                        usedRows.add(row)
                        usedCols.add(col)
            # compute both the row and column index we have NOT yet
            # examined
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)
            # in the event that the number of object centroids is
            # equal or greater than the number of input centroids
            # we need to check and see if some of these objects have
            # potentially disappeared
            if D.shape[0] >= D.shape[1]:
                # loop over the unused row indexes
                for row in unusedRows:
                    # grab the object ID for the corresponding row
                    # index and increment the disappeared counter
                    objectID = objectIDs[row]
                    self.missing_detection[objectID] = False

                    self.disappeared[objectID] += 1
                    self.continueMovement(objectID, verticalToleranceMovement)
                    self.pre_previousPos[objectID] = self.previousPos[objectID]
                    self.previousPos[objectID] = self.objects[objectID]
                    # check to see if the number of consecutive
                    # frames the object has been marked "disappeared"
                    # for warrants deregistering the object
                    # first two ifs check if vehicle is on edge of lane and use different maxdisappeared if so
                    if objectCentroids[row][1] < 280 and objectCentroids[row][0] < 50:
                        self.deregister(objectID)
                    elif objectCentroids[row][1] > 324 and objectCentroids[row][0] > 1230:
                        self.deregister(objectID)
                    else:
                        if self.disappeared[objectID] > self.maxDisappeared:
                            self.deregister(objectID)
            # otherwise, if the number of input centroids is greater
            # than the number of existing object centroids we need to
            # register each new input centroid as a trackable object
            else:
                for col in unusedCols:
                    if inputCentroids[col][1] < 277 and inputCentroids[col][0] >= 50:
                        self.register(inputCentroids[col])
                    if inputCentroids[col][1] > 327 and inputCentroids[col][0] <= 1230:
                        self.register(inputCentroids[col])
        # return the set of trackable objects
        return self.objects

    def continueMovement(self, objectID, verticalToleranceMovement):
        # check if values for the two previous positions are existent (i.e. the object has to be detected at least twice)
        if self.previousPos[objectID] is not None and self.pre_previousPos[objectID] is not None:
            print(prevMovement)
            # check if vertical movement is plausible
            if abs(prevMovement[1]) < verticalToleranceMovement:
                if (self.objects[objectID][1] < 302 and prevMovement[0]) < 0 or (self.objects[objectID][1] > 302 and prevMovement[0]) > 0:
                    self.objects[objectID] = self.objects[objectID] + self.previousPos[objectID] - self.pre_previousPos[objectID]
                    self.detection_status[self.nextObjectID] = True