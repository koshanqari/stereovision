import rospy
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats
from std_msgs.msg import Bool 
import numpy as np
import cv2
import matplotlib.pyplot as plt
from statistics import mean
import time 

rospy.init_node('depth_calculator', anonymous=True)

resolution_width = 1280 # pixels
resolution_height = 720 # pixels
resolution_channels =3


def get_frames(rostopic, t ):
	image =  Floats()
	image = rospy.wait_for_message(rostopic, numpy_msg(Floats))
	img = np.asarray(image.data, dtype='uint8')
	img = img.reshape(int(resolution_height), int(resolution_width), int(resolution_channels))
	return img
	


#Installing a trackbar to give live hsv data 
def nothing(x):
    pass
cv2.namedWindow("Trackbars")
cv2.createTrackbar("L - H", "Trackbars", 108, 360, nothing)
cv2.createTrackbar("L - S", "Trackbars", 149, 360, nothing)
cv2.createTrackbar("L - V", "Trackbars", 159, 360, nothing)
cv2.createTrackbar("U - H", "Trackbars", 255, 360, nothing)
cv2.createTrackbar("U - S", "Trackbars", 255, 360, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 360, nothing)

# NOTE ---> the values given for red might not be that of red, 
# as the pipeline ins configured for bgr not rgb


#kalman boy in the house------------------------------------------------------------------>>>>
import numpy as np
from pykalman import KalmanFilter
# Define the initial state and its covaience
# Assuming a constant velocity model
initial_state_mean = [0, 0, 0, 0, 0, 0]
initial_state_covariance=np.eye(6)


#Transition Matrix and its covarience 
transition_matrix = [[1, 0, 0, 0.25, 0, 0],
                     [0, 1, 0, 0, 0.25, 0],
                     [0, 0, 1, 0, 0, 0.25],
                     [0, 0, 0, 1, 0, 0],
                     [0, 0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 0, 1]]
# Define the process noise covariance matrix
process_covariance = np.eye(6) * 0.1



# Define the observation matrix and observation noise covariance matrix
observation_matrix = [[1, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0]]

observation_covariance = np.eye(3) * 0.01



# Create the Kalman Filter object
kf = KalmanFilter(transition_matrices=transition_matrix,
                  observation_matrices=observation_matrix,
                  initial_state_mean=initial_state_mean,
                  initial_state_covariance=initial_state_covariance,
                  observation_covariance=observation_covariance,
                  transition_covariance=process_covariance)

# Initialize the state estimate
state_estimate = initial_state_mean
state_covariance = initial_state_covariance
#-----------------------------------------------------------------------------------------





#we're defining then here to repair NameError: name 'center_x' is not defined
center_x_l = 0
center_y_l = 0
center_x_r = 0
center_y_r = 0
z = 0
k=0
import os
if __name__=="__main__":
	

	while True:
		start_time = time.time()
		image_right = get_frames('/camera_front_right', 'r')  
		image_left = get_frames('/camera_side_left', 'l')
		# cv2.imshow('Left', image_left)
		# cv2.imshow('Right', image_right)


		# <<--------------------------------------------------DOING THE THING FOR LEFT SIDE ------------------------------------------>>
		# converting frame(left image) to red frame --------------------------->>>
		frame = image_left #(get the frame from pipeline, ros etc, 3x3 matrix array)
		#converting to hsv -->>
		hsv_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
		#Creating a mask that masks everything except Red color
		l_h = cv2.getTrackbarPos("L - H", "Trackbars")
		l_s = cv2.getTrackbarPos("L - S", "Trackbars")
		l_v = cv2.getTrackbarPos("L - V", "Trackbars")
		u_h = cv2.getTrackbarPos("U - H", "Trackbars")
		u_s = cv2.getTrackbarPos("U - S", "Trackbars")
		u_v = cv2.getTrackbarPos("U - V", "Trackbars")
		lower_red = np.array([l_h, l_s, l_v])
		upper_red = np.array([u_h, u_s, u_v])
		mask_red = cv2.inRange(hsv_frame, lower_red, upper_red)
		#now putting the mask on the Frame
		red_frame = cv2.bitwise_and(frame, frame, mask=mask_red)
		# -------------------------------------------------------------------------
		

		#CENTROID: Finding the contours of the red obejct detected and ITS CENTROID--------------->>
		contours, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		if contours:
			# Find the sabsay bada contour
			largest_contour = max(contours, key=cv2.contourArea)
			# moments contour kay
			moments = cv2.moments(largest_contour)
			# Get the center of the contour (x, y)
			if (moments["m00"] != 0): #(if lagaya hay to repair the ZeroDivisionError: float division by zero)
				center_x_l = int(moments["m10"] / moments["m00"])
				center_y_l = int(moments["m01"] / moments["m00"])
				# z = depth_image[center_y, center_x]  
			# Drawing a circle x, y -->>
			cv2.circle(red_frame, (center_x_l, center_y_l), 5, (255, 255, 255), 2)
			cv2.circle(frame, (center_x_l, center_y_l), 5, (255, 255, 255), 2)
			cv2.line(frame, (center_x_l, 10000), (center_x_l, -10000), (0, 0,255), 2)
			cv2.line(frame, (10000, center_y_l), (-10000, center_y_l), (0, 0,255), 2)

			#drawing circles for alignment 
			for i in range(-600, 600, 50):
				cv2.circle(frame, (640+i, 360), 5, (0, 255, 0), 2)
				cv2.circle(frame, (640, 360+i), 5, (0, 255, 0), 2)
				cv2.putText(frame, f"{640+i}", (640+i, 350), fontScale=0.7, fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, color=(0,255,0))
			
			
		# --------------------------------------------------------------------------------------
			# <-----------RESULTS ---------->>>>>
			flip_frame = cv2.flip(frame, 1)
			cv2.imshow("LEFT Frame(flipped)", cv2.resize(flip_frame, (920, 500)))
			cv2.moveWindow("LEFT Frame(flipped)", 0, 50)
			cv2.imshow("LEFT resize red_frame", cv2.resize(red_frame, (920, 500)))
			# cv2.moveWindow("LEFT resize red_frame", 500, 500)
			# <----------------------------->>>>>
		# <<------------------------------------------------------END LEFT------------------------------------------------------------------->>
			







		# <<--------------------------------------------------DOING THE THING FOR RIGHT SIDE ------------------------------------------>>


		# converting frame(left image) to red frame --------------------------->>>
		frame = image_right #(get the frame from pipeline, ros etc, 3x3 matrix array)
		#converting to hsv -->>
		hsv_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
		#Creating a mask that masks everything except Red color
		l_h = cv2.getTrackbarPos("L - H", "Trackbars")
		l_s = cv2.getTrackbarPos("L - S", "Trackbars")
		l_v = cv2.getTrackbarPos("L - V", "Trackbars")
		u_h = cv2.getTrackbarPos("U - H", "Trackbars")
		u_s = cv2.getTrackbarPos("U - S", "Trackbars")
		u_v = cv2.getTrackbarPos("U - V", "Trackbars")
		lower_red = np.array([l_h, l_s, l_v])
		upper_red = np.array([u_h, u_s, u_v])
		mask_red = cv2.inRange(hsv_frame, lower_red, upper_red)
		#now putting the mask on the Frame
		red_frame = cv2.bitwise_and(frame, frame, mask=mask_red)
		# -------------------------------------------------------------------------
		

		#CENTROID: Finding the contours of the red obejct detected and ITS CENTROID--------------->>
		contours, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		if contours:
			# Find the sabsay bada contour
			largest_contour = max(contours, key=cv2.contourArea)
			# moments contour kay
			moments = cv2.moments(largest_contour)
			# Get the center of the contour (x, y)
			if (moments["m00"] != 0): #(if lagaya hay to repair the ZeroDivisionError: float division by zero)
				center_x_r = int(moments["m10"] / moments["m00"])
				center_y_r = int(moments["m01"] / moments["m00"])
				# z = depth_image[center_y, center_x]  
			# Drawing a circle x, y -->>
			cv2.circle(red_frame, (center_x_r, center_y_r), 5, (255, 255, 255), 2)
			cv2.circle(frame, (center_x_r, center_y_r), 5, (255, 255, 255), 2)
			cv2.line(frame, (center_x_r, 10000), (center_x_r, -10000), (0, 0,255), 2)
			cv2.line(frame, (10000, center_y_r), (-10000, center_y_r), (0, 0,255), 2)

			for i in range(-600, 600, 50):
				cv2.circle(frame, (640+i, 360), 5, (0, 255, 0), 2)
				cv2.circle(frame, (640, 360+i), 5, (0, 255, 0), 2)
				cv2.putText(frame, f"{640+i}", (640+i, 350), fontScale=0.7, fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, color=(0,255,0))
			
			
		# --------------------------------------------------------------------------------------
			# <-----------RESULTS ---------->>>>>
			flip_frame = cv2.flip(frame, 1)
			cv2.imshow("RIGHT Frame(flipped)", cv2.resize(flip_frame, (920, 500)))
			cv2.moveWindow("RIGHT Frame(flipped)", 1280, 50)
			cv2.imshow("RIGHT resize red_frame", cv2.resize(red_frame, (920, 500)))
			# <----------------------------->>>>>

		# <<------------------------------------------------------END RIGHT------------------------------------------------------------------->>
		


		#Printing Disparity AND FINAL COORDINATES -------------------------------------------->>>>>>
			
		fx = 966
		fy = 966
		cx = resolution_width/2
		cy = resolution_height/2
		base = 152

		# CALCULATION OF Z wrt camera COORDINATES 
		if(center_x_l-center_x_r != 0):
			z = (-1)*(base*(fx))/(center_x_l-center_x_r)
		# MOving avg
		z = (z + k)/2
		k = z
		# <-----------RESULTS FOR CAL of Z---------->>>>>
		print("\n\n\n\n")
		# print("LEFT EYE--> "+ str(center_x_l) + "px", str(center_y_l) + "px" + "     RIGHT EYE--> " + str(center_x_r) + "px", str(center_y_r) + "px" ) # z is in mm
		print(f"LEFT EYE-->  {str(center_x_l)} px, {str(center_y_l)} px      RIGHT EYE-->   {str(center_x_r)} px, {str(center_y_r)} px" ) # z is in mm
		px = (center_x_l + center_x_r)/2
		py = (center_y_l + center_y_r)/2
		print(f"AVG Calculated Coordinates in px:  {px} px {py} px {round(z)} mm {round(z)/10} cm")
		print('\n')
		# <<<<<<<<<<<<<------------Final Calculation of X, Y, Z IN WORLD COORDINATES----------------------->>>>>>>>>>>>
		Z = z - 450
		X = (center_x_l- cx)*z / fx
		Y = (-1)*(center_y_l- cy)*z / fy 
		#<<<<<<<<<<<<<<<<<<----------------RESTRICTION ---------------------------->>>>>>>>>>>



		#-------------------------------------------------------------------------------------


		print(f"Final Coordinates: X={round(X)}mm({round(X)/10})cm || Y={round(Y)}mm({round(Y)/10})cm || Z={round(Z)}mm({round(Z)/10})cm")


		# Kalman boy in the house --------------------->>>>>>>>
		# Get the latest x, y, z observation
		x, y, z = X, Y, Z
		observation = np.array([x, y, z])

		# Update the state estimate
		(state_estimate, state_covariance) = kf.filter_update(state_estimate, state_covariance, observation)
		
		# Use the state estimate for your purposes
		print(f"Kalman Coordinates: X={round(state_estimate[0])}mm({round(state_estimate[0])/10})cm || Y={round(state_estimate[1])}mm({round(state_estimate[1])/10})cm || Z={round(state_estimate[2])}mm({round(state_estimate[2])/10})cm")
	


		end_time = time.time()
		print("time taken by loop:" ,end_time-start_time)
		key = cv2.waitKey(1)
		if key == 113:
			break

		


		