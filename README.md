The implementation is for Python 2.7 using Linux as operating system.

The algorithm recieves real time data from the camera and tracks the persons eye momements and displays messages according to the eye position.

If there are more than one person identified in the frame the algorithm won't track any eye movements.

Before you can run the Mouse_eye_control.py file you have to make sure that you have all the
libreries installed and that you have the shape_predictor_68_face_landmarks.dat file downloaded from:
https://github.com/AKSHAYUBHAT/TensorFace/blob/master/openface/models/dlib/shape_predictor_68_face_landmarks.dat.

The required libraries are:

-cv2

-numpy

-dlib

-pynput

-time

-pandas
