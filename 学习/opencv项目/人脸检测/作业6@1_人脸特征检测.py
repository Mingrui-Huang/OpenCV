import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Image", (640, 480))

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=2, color=(0, 255, 0))

# static_image_mode = False,
# max_num_faces = 1,
# refine_landmarks = False,
# min_detection_confidence = 0.5,
# min_tracking_confidence = 0.5


pTime = 0
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    # print(results)
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, landmark_list=faceLms, connections=mpFaceMesh.FACEMESH_FACE_OVAL,
                                  landmark_drawing_spec=drawSpec)
            for id, lm in enumerate(faceLms.landmark):
                ih, iw, c = img.shape
                x, y = int(lm.x * iw), int(lm.y * ih)
                print(id, x, y)
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, "FPS:"+str(int(fps)), (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
    cv2.imshow("Image", img)
    cv2.waitKey(30)
