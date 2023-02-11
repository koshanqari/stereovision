#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import sys
import pyrealsense2 as rs
from rs_device_manager.realsense_device_manager_F_R import DeviceManager
import mediapipe as mp
import time
import pandas as pd

# FILE2 = open('S_L.txt', 'w')


import rospy
from rospy_tutorials.msg import Floats
from rospy.numpy_msg import numpy_msg


rospy.init_node('talker_front_right', anonymous=True)

pub = rospy.Publisher('/camera_front_right', numpy_msg(Floats), queue_size=10)


# #rospy.rate(10)

# #resetting the hardware first
# ctx = rs.context()
# devices = ctx.query_devices()
# for dev in devices:
#     dev.hardware_reset()

# In[2]:


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


# In[3]:

# resolution_width = 1280 # pixels
# resolution_height = 720 # pixels

resolution_width = 1280 # pixels
resolution_height = 720 # pixels



def show_images_wo_mp(frames_devices, write_show='publish'):
	"""
	Parameters:
	-----------
	frames_devices : dict
		The frames from the different devices
		keys: Tuple of (serial, product-line)
			Serial number and product line of the device
		values: [frame]
			frame: rs.frame()
				The frameset obtained over the active pipeline from the realsense device
				
	"""
	for (device_info, frame) in frames_devices.items():
		device = device_info[0] #serial number
		color_image = np.asarray(frame[rs.stream.color].get_data())
	
		if write_show=='publish':
			DATA_TO_PUBLISH  = color_image.reshape(resolution_height*resolution_width*3) 
			DATA_TO_PUBLISH = DATA_TO_PUBLISH.astype('float32')
			pub.publish(DATA_TO_PUBLISH)

		else:
			cv2.imshow('Color image from RealSense Device Nr: ' + device, color_image)
			##cv2.imwrite('Right.png',color_image)
			cv2.waitKey(1)


# In[24]:

x1 = 0
y1 = 0

def show_images_with_mp(frames_devices, hands, draw=False, write_show='publish'):
	global x1
	global y1
	global resolution_width
	global resolution_height
	"""
	Parameters:
	-----------
	frames_devices : dict
		The frames from the different devices
		keys: Tuple of (serial, product-line)
			Serial number and product line of the device
		values: [frame]
			frame: rs.frame()
				The frameset obtained over the active pipeline from the realsense device
				
	"""
	for (device_info, frame) in frames_devices.items():
		device = device_info[0] #serial number
		color_image = np.asarray(frame[rs.stream.color].get_data())
		color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
		#print(color_image.shape)	
		#color_image.flags.writeable = False
		results = hands.process(color_image)
		color_image.flags.writeable = True
		if results.multi_hand_landmarks:
			for hand_landmarks in results.multi_hand_landmarks:
				#print(type(hand_landmarks))
				if draw == True:
					mp_drawing.draw_landmarks(color_image,hand_landmarks,mp_hands.HAND_CONNECTIONS,mp_drawing_styles.get_default_hand_landmarks_style(),mp_drawing_styles.get_default_hand_connections_style())	
				x1,y1=hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * resolution_width, hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * resolution_height
				#print(x1,y1)

				#cv2.circle(color_image, (int(x1),int(y1)), 10, (255, 0, 0) , -1)
				
				# stri = (str(x1)+', '+str(y1)+'  {} \n'.format(time.time()))
				# FILE2.write(stri)
				#coor_Left= pd.DataFrame()
				#coor_Left['X'] = [int(x1)]
				#coor_Left['Y'] = [int(y1)]

				# print(coor_Left)
				#coor_Left.to_csv('Left_coordinates.csv', mode='a', index=None, header=None)
			
		#print(color_image.shape)

		if write_show=='write':
			cv2.imwrite('Left.png', color_image)
		elif write_show =='publish':
			#print(color_image.shape)
			#print(int(x1),int(y1))

			color_image_flattened = color_image.reshape(resolution_height*resolution_width*3)  #3 for R,G,B channels
			#color_image_flattened = color_image_flattened.astype('uint8')


			color_image_specs = np.array([resolution_height,resolution_width,3,int(x1), int(y1)])	
			#color_image_specs = color_image_specs.astype('float32')
			#print(color_image_specs)
			DATA_TO_PUBLISH = np.concatenate((color_image_flattened,color_image_specs), axis=None)
			DATA_TO_PUBLISH = DATA_TO_PUBLISH.astype('float32')
			#print(DATA_TO_PUBLISH[0:100])
			#data_to_send = np.array([0,0], dtype=np.float32)
			#cv2.imshow('Color image from RealSense Device Nr: ' + device, color_image)
			pub.publish(DATA_TO_PUBLISH)
			#print('Sent')
			#cv2.waitKey(1)
			
		else:
			cv2.imshow('Color image from RealSense Device Nr: ' + device, color_image)
			cv2.waitKey(1)
		#cv2.imwrite('Left.png', color_image)
		#data_to_send = np.array([0,0], dtype=np.float32)
		#pub.publish(data_to_send)
		#cv2.waitKey(1)


# In[25]:


frame_rate = 15  # fps

dispose_frames_for_stablisation = 30  # frames


# In[26]:


left_camera_config = rs.config()

left_camera_config.enable_stream(rs.stream.color, resolution_width, resolution_height, rs.format.bgr8, frame_rate)

# Use the device manager class to enable the devices and get the frames
device_manager = DeviceManager(rs.context(), left_camera_config)
device_manager.enable_all_devices()
#device_manager.enable_device(device_manager._available_devices)



# In[27]:


#list(device_manager._available_devices.keys()) #devices enabled and connected


# In[28]:
print(device_manager._available_devices)
print('\nConnected Devices:'+str(len(device_manager._available_devices)))
assert( len(device_manager._available_devices) > 0 ) # at least one connected and enabled
device_manager.enable_emitter(False)


# In[ ]:


#device_manager.get_device_intrinsics(frames) # get intrinsic details of devices


# In[ ]:





# In[ ]:


#device_manager.load_settings_json("./HighResHighAccuracyPreset.json")


# In[20]:


with mp_hands.Hands(model_complexity=0,min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=1) as hands:
	try:
		while True:
			#Get stream from camera(s)
			frames_devices = device_manager.poll_frames()
			#SHOW THEM WHAT YOU GOT!#SHOW THEM WHAT YOU GOT!
			show_images_wo_mp(frames_devices)
			#media pipe!!!
			#show_images_with_mp(frames_devices, hands)

	except KeyboardInterrupt:
		print("The program was interupted by the user. Closing the program...")
	finally:
		device_manager.disable_streams()
		cv2.destroyAllWindows()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




