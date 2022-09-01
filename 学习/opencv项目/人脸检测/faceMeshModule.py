import cv2
import mediapipe as mp
import time

import numpy as np


class FaceMeshDetector():

    def __init__(self, static_image_mode=False,
                 max_num_faces=2,
                 refine_landmarks=True,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):

        self.static_image_mode = static_image_mode
        self.refine_landmarks = refine_landmarks
        self.max_num_faces = max_num_faces
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(static_image_mode=self.static_image_mode,
                                                 max_num_faces=self.max_num_faces,
                                                 refine_landmarks=self.refine_landmarks,
                                                 min_detection_confidence=self.min_detection_confidence,
                                                 min_tracking_confidence=self.min_tracking_confidence)

        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))

    def findFaceMesh(self, img, draw=True):

        ih, iw, c = img.shape
        canvas = np.zeros((ih, iw, 3), dtype=np.uint8)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(imgRGB)

        faces = []
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img,
                                               landmark_list=faceLms,
                                               connections=self.mpFaceMesh.FACEMESH_FACE_OVAL,
                                               landmark_drawing_spec=self.drawSpec)
                    self.mpDraw.draw_landmarks(canvas,
                                               landmark_list=faceLms,
                                               connections=self.mpFaceMesh.FACEMESH_FACE_OVAL,
                                               landmark_drawing_spec=self.drawSpec)
                face = []
                for id, lm in enumerate(faceLms.landmark):
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    # cv2.putText(img, str(id), (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
                    face.append([x, y])
                faces.append(face)
        return img, canvas, faces


def main():
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Image", (1280, 400))

    faceDetector = FaceMeshDetector()
    pTime = 0
    while True:
        success, img = cap.read()
        img, canvas, faces = faceDetector.findFaceMesh(img, draw=True)
        # if len(faces) != 0:
        # print(len(faces[0]), faces[0])
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(canvas, "FPS:" + str(int(fps)), (20, 40), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
        cv2.imshow("Image", np.hstack((img, canvas)))
        if cv2.waitKey(30) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
