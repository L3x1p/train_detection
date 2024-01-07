import cv2 as cv
import numpy as np

class YOLODetector:
    def __init__(self, modelConf="train.cfg", modelWeights="train_last.weights", classesFile="obj.names", confThreshold=0.5, nmsThreshold=0.1, inpWidth=416, inpHeight=416):
        self.confThreshold = confThreshold
        self.nmsThreshold = nmsThreshold
        self.inpWidth = inpWidth
        self.inpHeight = inpHeight
        self.classes = None
        with open(classesFile, 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')
        self.net = cv.dnn.readNetFromDarknet(modelConf, modelWeights)
        self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
        self.intersection_count = 0
        self.no_detections_count = 0  # Counts frames without detections


    def getOutputsNames(self):
        layersNames = self.net.getLayerNames()
        return [layersNames[i - 1] for i in self.net.getUnconnectedOutLayers()]

    def postprocess(self, frame, outs, interest_area):
        frameHeight, frameWidth = frame.shape[:2]
        # print(frameHeight, frameWidth)
        classIDs, confidences, boxes = [], [], []
        intersection = False

        for out in outs:
            for detection in out:
                # print(detection)
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                if confidence > self.confThreshold:
                    centerX, centerY = int(detection[0] * frameWidth), int(detection[1] * frameHeight)
                    width, height = int(detection[2] * frameWidth), int(detection[3] * frameHeight)
                    left, top = int(centerX - width / 2), int(centerY - height / 2)
                    boxes.append([left, top, width, height])
                    classIDs.append(classID)
                    confidences.append(float(confidence))

        indices = cv.dnn.NMSBoxes(boxes, confidences, self.confThreshold, self.nmsThreshold)
        for i in indices:
            box = boxes[i]
            self.drawPred(frame, classIDs[i], confidences[i], *box)
            if self.check_intersection(box, interest_area):
                intersection = True

        if intersection:
            self.intersection_count += 1
        if intersection:
            self.intersection_count += 1
            self.no_detections_count = 0  # Reset count as there's an intersection
        else:
            self.no_detections_count += 1
            if self.no_detections_count > 10:
                self.intersection_count = 0  # Reset intersection count if no detections for more than 10 frames

        return frame, intersection

    def drawPred(self, frame, classId, conf, left, top, width, height):
        right, bottom = left + width, top + height
        # Draw a bounding box.
        cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)

        label = '%.2f' % conf

        # Get the label for the class name and its confidence
        if self.classes:
            assert (classId < len(self.classes))
            label = '%s:%s' % (self.classes[classId], label)

        labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        cv.rectangle(frame, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine),
                     (255, 255, 255), cv.FILLED)
        cv.rectangle(frame, (left, top), (right, bottom), (255, 255, 255), 1)
        cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)

    def check_intersection(self, box, interest_area):
        # Extracting coordinates for the object box
        ax1, ay1, ax2, ay2 = box[0], box[1], box[0] + box[2], box[1] + box[3]

        # Extracting coordinates for the AOI
        bx1, by1, bx2, by2 = interest_area

        # Check if bx2 and by2 are given as width and height
        if bx2 < bx1 or by2 < by1:
            # Convert them to right and bottom coordinates
            bx2 += bx1
            by2 += by1

        # Check if box and interest_area intersect
        intersect = ax1 < bx2 and ax2 > bx1 and ay1 < by2 and ay2 > by1
        return intersect

    def process_frame(self, frame, interest_area):
        blob = cv.dnn.blobFromImage(frame, 1 / 255, (self.inpWidth, self.inpHeight), [0, 0, 0], 1, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.getOutputsNames())
        processed_frame, intersection = self.postprocess(frame, outs, interest_area)
        return processed_frame, intersection, self.intersection_count