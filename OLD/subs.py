import rospy
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats
from std_msgs.msg import Bool 
import numpy as np
import cv2
import matplotlib.pyplot as plt

rospy.init_node('depth_calculator', anonymous=True)

resolution_width = 1280 # pixels
resolution_height = 720 # pixels
resolution_channels =3


def get_frames(rostopic, t ):
	image =  Floats()
	image = rospy.wait_for_message(rostopic, numpy_msg(Floats))
	#cv2.imshow('Left',)
	#image = image[:-5]
	#print(t)
	#print(image.data[-5:])

	img = np.asarray(image.data, dtype='uint8')
	#print(img.shape)
	img = img.reshape(int(resolution_height), int(resolution_width), int(resolution_channels))
	return img
	



import os
if __name__=="__main__":

	while True:
		image_right = get_frames('/camera_front_right', 'r')  #front right is malfunctioning??? delayed publising!
		
		image_left = get_frames('/camera_side_left', 'l')
		#image_right = get_frames('/camera_front_right')  #front right is malfunctioning??? delayed publising!
		#cv2.imwrite('./left_camera_checkerboard/Checkerboard_Left_'+str(idx)+'.png', checkerboard_left)
		#cv2.imwrite('./right_camera_checkerboard/Checkerboard_Right_'+str(idx)+'.png', checkerboard_right)
		#cv2.waitKey(1)
		cv2.imshow('Left', image_left)
		cv2.imshow('Right', image_right)



		#image_left is the RGB image -> Segment it for red color
		cv2.waitKey(1)
		#idx = idx + 1

		