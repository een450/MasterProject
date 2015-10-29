# MasterProject
Automatic camera selection mechanism based on facial feature detection

This tool is implemented in Python, with the library OpenCV (http://opencv.org/). 

The file dualwebcam.py is the implementation of the decisionmodel of the two webcams based on face detection.
The haarcascade .xml files are the classifiers used for frontal face detection and other facial feautures:
    - frontalface_alt2.xml has a slightly better performance on frontal face detection than frontalface_default.xml
    - eye_tree_eyeglasses.xml for detection of the eyes with glasses
    - mcs_mouth.xml for mouth detection
    - mcs_nose.xml for nose detection



Interested in getting started with face detection in Python? Try:
https://realpython.com/blog/python/face-recognition-with-python/ for "Face recognition in Pyhon under 25 lines of code".
