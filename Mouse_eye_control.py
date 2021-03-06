import cv2
import numpy as np
import dlib
from math import hypot
from pynput.mouse import Button, Controller
import time
import pandas as pd
import numpy as np

"""
In order to make this script run you first need download a file named:
shape_predictor_68_face_landmarks.dat
You can find this file here:
https://github.com/AKSHAYUBHAT/TensorFace/blob/master/openface/models/dlib/shape_predictor_68_face_landmarks.dat
"""

# Getting a refference to the main camera
cap = cv2.VideoCapture(0)

# Getting a refference to the mouse, and the downloaded file
mouse = Controller()
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

font = cv2.FONT_HERSHEY_PLAIN


# This function will adjust the gamma of the current frame in order to make the frame more easy to interpretate
def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    return cv2.LUT(image, table)


def midpoint(p1, p2):
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)


# This function calculates a ratio that is used to determine if a user is blinking or not
def get_blinking_ratio(eye_points, facial_landmarks):
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))
    hor_line_lenght = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_lenght = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))

    ratio = hor_line_lenght / ver_line_lenght
    return ratio


# In this function we determine if the user is looking left or right
def get_gaze_ratio(eye_points, facial_landmarks):
    gray_eye, color_eye = eye_seting(eye_points, facial_landmarks)
    _, threshold_eye = cv2.threshold(gray_eye, 100, 255, cv2.THRESH_BINARY)

    height, width = threshold_eye.shape
    left_side_threshold = threshold_eye[0: height, 0: width // 2]
    left_side_white = cv2.countNonZero(left_side_threshold)
    cv2.imshow("Threshold", threshold_eye)
    right_side_threshold = threshold_eye[0: height, width // 2: width]
    right_side_white = cv2.countNonZero(right_side_threshold)
    if left_side_white == 0:
        gaze_ratio = 1
    elif right_side_white == 0:
        gaze_ratio = 5
    else:
        gaze_ratio = left_side_white / right_side_white
    return gaze_ratio, left_side_white, right_side_white


# In this function we determine if the user is looking up or down
def get_up_down(eye_points, facial_landmarks):
    gray_eye, color_eye = eye_seting(eye_points, facial_landmarks)

    _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
    _, threshold_color_eye = cv2.threshold(color_eye, 70, 255, cv2.THRESH_BINARY)
    gray_thresh = cv2.cvtColor(threshold_color_eye, cv2.COLOR_BGR2GRAY)
    _, threshold_color_eye2 = cv2.threshold(gray_thresh, 250, 255, cv2.THRESH_BINARY)

    summ = cv2.countNonZero(threshold_eye)
    summ_color = cv2.countNonZero(threshold_color_eye2)
    height, width = threshold_eye.shape
    up_side_threshold = threshold_eye[0: height // 2, 0: width]
    up_side_threshold_color = threshold_color_eye2[height // 5: int(height / (1.5)), width // (4): int(width // (1.5))]
    value = cv2.countNonZero(up_side_threshold_color)
    up_side_white = cv2.countNonZero(up_side_threshold)
    down_side_threshold = threshold_eye[height // 2: height, 0: width]
    down_side_white = cv2.countNonZero(down_side_threshold)
    cv2.imshow("Threshold_gray", threshold_color_eye)
    return summ, up_side_white, down_side_white, value, summ_color


# The function makes a mask around the eyes that contains only the iris and the white part
def eye_seting(eye_points, facial_landmarks):
    # The eye region from the facial landmarks previously placed on the user face is extracted
    eye_region = np.array([(facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
                           (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
                           (facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
                           (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
                           (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),
                           (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)], np.int32)
    # From the current frame we extract the parts that we don't need
    height, width, _ = frame.shape
    mask = np.zeros((height, width), np.uint8)
    cv2.polylines(mask, [eye_region], True, 255, 2)
    cv2.fillPoly(mask, [eye_region], 255)
    eye = cv2.bitwise_and(gray, gray, mask=mask)
    color_eye = cv2.bitwise_and(frame, frame, mask=mask)

    min_x = np.min(eye_region[:, 0])
    max_x = np.max(eye_region[:, 0])
    min_y = np.min(eye_region[:, 1])
    max_y = np.max(eye_region[:, 1])

    gray_eye = eye[min_y: max_y, min_x: max_x]
    color_eye = color_eye[min_y: max_y, min_x: max_x]
    return gray_eye, color_eye


cont = 0
start = 0
end = 0
# A loop that will stop only when the user presses the "q" key
while True:
    # Get the input from the camera
    _, frame = cap.read()
    # The input is processed to obtain better results
    new_frame = np.zeros((500, 500, 3), np.uint8)
    gray = cv2.cvtColor(adjust_gamma(frame, 1.1), cv2.COLOR_BGR2GRAY)
    gamma_adjusted = adjust_gamma(frame, 1.5)
    # The detectors counts all the faces seen in the frame
    faces = detector(gray)
    for face in faces:
        # In order to obtain consistent results the program analyses only a face and does not proceed is more are found
        if len(faces) == 1:
            # Place the landmarks on the face
            landmarks = predictor(gray, face)
            # Check if the user is blinking
            left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
            right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)
            blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2
            if blinking_ratio > 5.7:
                cv2.putText(frame, "BLINKING", (50, 150), font, 7, (255, 0, 0))
                if cont % 2 == 0:
                    start = time.clock()
                if cont % 2 == 1:
                    end = time.clock()
                # If the user blinks in quick succession the message "Double click is printed in the console"
                if end - start < 1.5 and end - start > 0.1 and cont % 2 == 1:
                    print("Double Click")
                    cont = -1
                elif cont % 2 == 1:
                    cont = -1
                cont += 1
            # Get the direction in which the user is looking
            gaze_ratio_left_eye, llw, lrw = get_gaze_ratio([36, 37, 38, 39, 40, 41], landmarks)
            gaze_ratio_right_eye, rlw, rrw = get_gaze_ratio([42, 43, 44, 45, 46, 47], landmarks)
            summ1, ldw, luw, val1, summ21 = get_up_down([36, 37, 38, 39, 40, 41], landmarks)
            summ2, rdw, ruw, val2, summ22 = get_up_down([42, 43, 44, 45, 46, 47], landmarks)
            if rrw + lrw == 0:
                rrw = 1
            gaze_ratio = (llw + rlw) / (rrw + lrw)
            # Show a message on the created frame which contains the direction in which the user is looking
            if (rrw + lrw) * (0.75) > (llw + rlw):
                cv2.putText(frame, "RIGHT", (50, 100), font, 2, (0, 0, 255), 3)
                new_frame[:] = (0, 0, 255)
                mouse.move(15, 0)
            elif (llw + rlw) * (0.75) > (rrw + lrw):
                new_frame[:] = (255, 0, 0)
                cv2.putText(frame, "LEFT", (50, 100), font, 2, (0, 0, 255), 3)
                mouse.move(-15, 0)
            else:
                new_frame[:] = (0, 0, 0)
                cv2.putText(frame, "CENTER", (50, 100), font, 2, (0, 0, 255), 3)
            if summ1 + summ2 == 0:
                summ1 = 1
            if val1 + val2 == 0:
                val1 = 1
            ratio = float((summ21 + summ22)) / (val1 + val2)
            if ratio <= 6:
                cv2.putText(frame, "Up", (50, 200), font, 2, (0, 0, 255), 3)
                new_frame[:] = (0, 0, 255)
                mouse.move(0, -15)
            elif 6 < ratio and ratio < 8.5:
                new_frame[:] = (255, 0, 0)
                cv2.putText(frame, "Center", (50, 200), font, 2, (0, 0, 255), 3)
            else:
                new_frame[:] = (0, 0, 0)
                cv2.putText(frame, "Down", (50, 200), font, 2, (0, 0, 255), 3)
                mouse.move(0, 15)
    # Show the images that are of interest
    cv2.imshow("Frame", frame)
    cv2.imshow("New frame", new_frame)
    cv2.imshow("Gamma_adjusted", gamma_adjusted)
    key = cv2.waitKey(1)
    if key == 113:
        break

# Close all created windows and close the camera
cap.release()
cv2.destroyAllWindows()
