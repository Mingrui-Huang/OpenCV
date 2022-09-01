import cv2
import math
import mediapipe as mp
import time
import numpy as np


class handDetector():
    def __init__(self, mirror=False, mode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplexity = modelComplexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
                                        self.mode,
                                        self.maxHands,
                                        self.modelComplexity,
                                        self.detectionCon,
                                        self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

        self.mirror = mirror
        self.tipIds = [4, 8, 12, 16, 20]
        self.fingers = []
        self.landmarksList = []

    def findHands(self, frame, draw=True):

        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(frameRGB)

        if self.results.multi_hand_landmarks:
            for handlandmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                                                frame,
                                                handlandmarks,
                                                self.mpHands.HAND_CONNECTIONS)
        return frame

    def findPosition(self, frame, handNo=0, draw=True):

        xList = []
        yList = []
        bbox = []
        self.landmarksList = []
        if self.results.multi_hand_landmarks:
            myhand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myhand.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                # print(id, cx, cy)
                self.landmarksList.append([id, cx, cy])
                if draw:
                    cv2.circle(frame, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = [xmin, ymin, xmax, ymax]
            if draw:
                cv2.rectangle(frame, (xmin-20, ymin-20), (xmax+20, ymax+20), (255, 0, 255), 2)
        return self.landmarksList, bbox

    def findDistance(self, p1, p2, img, draw=True, r=15, t=3):
        p1 = self.landmarksList[p1][1:]
        p2 = self.landmarksList[p2][1:]
        length = math.hypot(p1[0] - p2[0], p1[1] - p2[1])
        cx, cy = (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2
        center = (cx, cy)
        img = img
        if draw:
            cv2.line(img, p1, p2, (255, 0, 255), t)
            cv2.circle(img, center, r, (255, 0, 255), cv2.FILLED)

            return length, img, center

    def handtype(self):
        if self.results.multi_hand_landmarks:
            if not self.mirror:
                if self.landmarksList[17][1] < self.landmarksList[5][1]:
                    return "right"
                else:
                    return "left"
            else:
                if self.landmarksList[17][1] < self.landmarksList[5][1]:
                    return "left"
                else:
                    return "right"
    def fingersUp(self):
        if self.results.multi_hand_landmarks:
            myHandType = self.handtype()
            # print(myHandType)
            fingers = []
            # Thumb
            if not self.mirror:
                if myHandType == "right":
                    if self.landmarksList[self.tipIds[0]][1] > self.landmarksList[self.tipIds[0] - 1][1]:
                        fingers.append(1)
                    else:
                        fingers.append(0)
                else:
                    if self.landmarksList[self.tipIds[0]][1] < self.landmarksList[self.tipIds[0] - 1][1]:
                        fingers.append(1)
                    else:
                        fingers.append(0)
            if self.mirror:
                if myHandType == "right":
                    if self.landmarksList[self.tipIds[0]][1] < self.landmarksList[self.tipIds[0] - 1][1]:
                        fingers.append(1)
                    else:
                        fingers.append(0)
                else:
                    if self.landmarksList[self.tipIds[0]][1] > self.landmarksList[self.tipIds[0] - 1][1]:
                        fingers.append(1)
                    else:
                        fingers.append(0)
            # 4 Fingers
            for id in range(1, 5):
                if self.landmarksList[self.tipIds[id]][2] < self.landmarksList[self.tipIds[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            return fingers

def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector(mirror=False)
    while True:
        success, frame = cap.read()
        # frame = cv2.flip(frame, 1)
        frame = detector.findHands(frame)
        landmarksList = detector.findPosition(frame)

        if len(landmarksList) != 0:
            handtype = detector.handtype()
            fingers = detector.fingersUp()
            if fingers is not None:
                totalFingers = fingers.count(1)
            else:
                totalFingers = None
            print(handtype, totalFingers)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(frame, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

        cv2.imshow('frame_originaql', frame)
        if cv2.waitKey(1) == ord('q'):
            break

if __name__=="__main__":
    main()