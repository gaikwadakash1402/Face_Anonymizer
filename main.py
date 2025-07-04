import cv2
import mediapipe as mp


# Read Image
img = cv2.imread("C:/Users/user/Computer_Vision/ID_Photo.jpg")

H, W, _ = img.shape


# Detect Faces
mp_face_detection = mp.solutions.face_detection

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = face_detection.process(img_rgb)

    # print(out.detections)
    for detection in out.detections:
        location_data = detection.location_data

        bbox = location_data.relative_bounding_box

        x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height

        x1, y1, w, h = int(x1 * W), int(y1 * H), int(w * W), int(h * H)

        # Blurr Faces
        img[y1:y1 + h, x1:x1 + w] = cv2.GaussianBlur(img[y1:y1 + h, x1:x1 + w], (31, 31), 30)

        cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), (0, 0, 255), 2)


    # cv2.imshow("Image", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows() 

# Save Image
cv2.imwrite("C:/Users/user/Computer_Vision/Face_Anonymizer/Output_Images/ID_Photo_Blurred.jpg", img)

