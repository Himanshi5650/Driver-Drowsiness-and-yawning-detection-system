from scipy.spatial import distance
from imutils import face_utils
from pygame import mixer
import imutils
import dlib
import cv2


mixer.init()
mixer.music.load(r"C:\Users\SUNNY\Downloads\music.wav")


def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[14], mouth[18])
    B = distance.euclidean(mouth[12], mouth[16])
    mar = A / B
    return mar


thresh_eye = 0.25
thresh_mouth = 0.6  # Adjust as needed
frame_check = 20

# Initialize face detector and facial landmark predictor
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor(r"C:\Users\SUNNY\Downloads\shape_predictor_68_face_landmarks.dat")

# Indices for left and right eyes and mouth in the facial landmark coordinates
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

# Start video capture from the default camera
cap = cv2.VideoCapture(0)

# Flags to track consecutive frames where drowsiness or yawning is detected
flag_eye = 0
flag_mouth = 0

# Main loop for processing each frame
while True:
    # Read frame from the video stream
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=550)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    subjects = detect(gray, 0)
    
    # Loop through detected faces
    for subject in subjects:
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)
        left_eye = shape[lStart:lEnd]
        right_eye = shape[rStart:rEnd]
        mouth = shape[mStart:mEnd]
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0
        mouth_ratio = mouth_aspect_ratio(mouth)

        left_eye_hull = cv2.convexHull(left_eye)
        right_eye_hull = cv2.convexHull(right_eye)
        mouth_hull = cv2.convexHull(mouth)

        cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [mouth_hull], -1, (0, 255, 0), 1)

        # Check for drowsiness
        if ear < thresh_eye:
            flag_eye += 1
            if flag_eye >= frame_check:
                cv2.putText(frame, "****************SLEEPING ALERT!!****************", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "****************SLEEPING ALERT!****************", (10, 325),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                mixer.music.play()
        else:
            flag_eye = 0

        # Check for yawning
        if mouth_ratio > thresh_mouth:
            flag_mouth += 1
            if flag_mouth >= frame_check:
                cv2.putText(frame, "****************YOU ARE YAWNING!!****************", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                mixer.music.play()
        else:
            flag_mouth = 0

    # Display the processed frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Clean up
cv2.destroyAllWindows()
cap.release()
