import numpy as np
import cv2
import sys

# Cascade classifiers
#face_cascade = cv2.CascadeClassifier('C:/Users/Eveline/Documents/Thesis/Camera detection/Face detection programming/Webcam-Face-Detect-master/haarcascade_frontalface_alt2.xml')
#eye_cascade = cv2.CascadeClassifier('C:/Users/Eveline/Documents/Thesis/Camera detection/Face detection programming/Webcam-Face-Detect-master/haarcascade_eye.xml')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# video capture
cap = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1)

# sensitivity
TIMER = 15

def faceDetect(frame):
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray,1.3,6)

	# Search within detected frontal face area for facial features (eyes)
	for (x, y, w, h) in faces:
		#cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = frame[y:y+h, x:x+w]

		eyes = eye_cascade.detectMultiScale(roi_gray,2,6, 1)

		# draw rectangle around eyes
		#for (ex,ey,ew,eh) in eyes:
			#cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

		#return list of detected faces and eyes
		return np.array(faces).tolist(), len(eyes), np.array(eyes).tolist()

# Keep track of detected faces in total
def compareCameras(c_main,c_side):
	global detected
	if c_main != None and c_side != None:
		detected.append('both')
	elif c_main != None:
		detected.append('main')
	elif c_side != None:
		detected.append('side')
	elif c_main == None and c_side == None:
		detected.append('none')

	detected = detected[1:]  # keep detect list up to certain amount of detected faces in history. Append new item, then remove first (old) item.


# Check if system is allowed to switch view
def switchAllowed():
	global detected, choosen, TIMER
	count = 0
	for i in range(0,TIMER):        # Check if previously chosen webcam views are similar
		if choosen[-i] == choosen[-i-1]:
			count += 1

	if count == TIMER:				# Allowed to switch when items in list of previously choosen webcams are all similar
		return  'allowed'
	else: return 'not allowed'


# Select webcam view
def chooseCamera(switch):
	global detected, choosen
	if detected[-1] == 'both' or detected[-1] == 'none':
		show = choosen[-1]			# show previously choosen camera
	elif switch == 'allowed':
		show = detected[-1]			# show actually detected face
	elif switch =='not allowed':
		show = choosen[-1]			# show previously choosen camera

	choosen.append(show)
	choosen = choosen[1:]

	return show

# Add description of detected and showed webcam view in interface
def featureComments(camera,main,side):
	global detected
	font = cv2.FONT_HERSHEY_SIMPLEX
	cv2.putText(main,('detected  '+ str(detected[-1])),(10,100), font, 0.75,(0,0,255),2,cv2.LINE_AA)
	cv2.putText(side,('detected  ' + str(detected[-1])),(10,100), font, 0.75,(0,0,255),2,cv2.LINE_AA)
	cv2.putText(main,('show '+ str(camera)),(10,130), font, 0.75,(255,0,255),2,cv2.LINE_AA)
	cv2.putText(side,('show  ' + str(camera)),(10,130), font, 0.75,(255,0,255),2,cv2.LINE_AA)
	cv2.putText(main,('eyes detected ' + str(numbereyes)),(10,160), font, 0.75,(255,0,0),2,cv2.LINE_AA)


while (cv2.waitKey(1) & 0xFF != ord('q')):
	ret, main = cap.read()
	ret, side = cap2.read()

	features_main = faceDetect(main)			# feature detection; face & eyes
	features_side = faceDetect(side)

	compareCameras(features_main,features_side) # keep track of detected faces on cameras
	switch = switchAllowed()      				# check if switching is allowed
	camera = chooseCamera(switch)				# choose camera side based on previous switching, save choosen camera

	#featureComments(camera,main,side)			# show comments on detected features

	if camera == 'main':
		cv2.imshow('both', main)

	elif camera == 'side':
		cv2.imshow('both', side)

	# Show both webcam feeds in separate windows
	#cv2.imshow('frontal', main)
	#cv2.imshow('side', side)

cap.release()
cv2.destroyAllWindows()
