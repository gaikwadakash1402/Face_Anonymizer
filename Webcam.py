import cv2
import mediapipe as mp

# Initialize mediapipe face detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)

# Set webcam resolution (optional)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(frame_rgb)

        # If face(s) detected, blur them
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x1 = int(bbox.xmin * w)
                y1 = int(bbox.ymin * h)
                x2 = x1 + int(bbox.width * w)
                y2 = y1 + int(bbox.height * h)

                # Ensure coordinates are within the frame
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                # Extract face ROI and apply Gaussian Blur
                face_roi = frame[y1:y2, x1:x2]
                blurred_face = cv2.GaussianBlur(face_roi, (99, 99), 30)
                frame[y1:y2, x1:x2] = blurred_face

        cv2.imshow('Face Anonymizer - Press ESC to Exit', frame)

        # Exit on ESC key
        if cv2.waitKey(5) & 0xFF == 27:

            break

cap.release()
cv2.destroyAllWindows()

