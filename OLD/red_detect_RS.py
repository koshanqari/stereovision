import cv2
import numpy as np
import pyrealsense2 as rs

# def nothing(x):
#     pass

# webcam = cv2.VideoCapture(0)


# # Taking input from the realsense---->
# #resetting the hardware first
# ctx = rs.context()
# devices = ctx.query_devices()
# for dev in devices:
#     dev.hardware_reset()


# # Create a pipeline
# pipeline = rs.pipeline()
# config = rs.config()
# config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
# config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
# # Start the pipeline
# pipeline.start(config)
# ---------------

#Installing a trackbar to give live hsv data 
# cv2.namedWindow("Trackbars")
# cv2.createTrackbar("L - H", "Trackbars", 114, 360, nothing)
# cv2.createTrackbar("L - S", "Trackbars", 127, 360, nothing)
# cv2.createTrackbar("L - V", "Trackbars", 76, 360, nothing)
# cv2.createTrackbar("U - H", "Trackbars", 255, 360, nothing)
# cv2.createTrackbar("U - S", "Trackbars", 255, 360, nothing)
# cv2.createTrackbar("U - V", "Trackbars", 255, 360, nothing)

# # NOTE ---> the values given for red might not be that of red, 
# # as the pipeline ins configured for bgr not rgb


#we're defining then here to repair NameError: name 'center_x' is not defined
center_x = 0
center_y = 0
z = 0
while True:
    
    #Taking the frame from wbcam-->>
    # _, frame_wc = webcam.read()
    # cv2.imshow("Frame", frame_wc)


    # Taking input from the pipeline----------------->>
    # Wait for a frame
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    # Convert the frames to numpy arrays
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())
    # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET) # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
    #OUTPUT aata hay hamko = color_image & depth_colormap
    # --------------------------


    frame = color_image
    #converting to hsv -->>
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    # cv2.imshow("hsv_Frame", hsv_frame)
    #Creating a mask that masks everything except Red color --->>>
    l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    u_v = cv2.getTrackbarPos("U - V", "Trackbars")
    lower_red = np.array([l_h, l_s, l_v])
    upper_red = np.array([u_h, u_s, u_v])
    mask_red = cv2.inRange(hsv_frame, lower_red, upper_red)
    # cv2.imshow("mask_red", mask_red)
    #now putting the mask on the Frame
    red_frame = cv2.bitwise_and(frame, frame, mask=mask_red)
    # cv2.imshow("red_frame", red_frame)


    key = cv2.waitKey(1)
    if key == 113:
        break


    # Find the contours of the red band
    contours, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find the sabsay bada contour
        largest_contour = max(contours, key=cv2.contourArea)

        # moments contour kay
        moments = cv2.moments(largest_contour)

        # Get the center of the contour (x, y)
        if (moments["m00"] != 0): #(if lagaya hay to repair the ZeroDivisionError: float division by zero)
            
            center_x = int(moments["m10"] / moments["m00"])
            center_y = int(moments["m01"] / moments["m00"])
            z = depth_image[center_y, center_x]  
        

        # Drawing a circle x, y -->>
        cv2.circle(red_frame, (center_x, center_y), 5, (255, 255, 255), 2)
        cv2.circle(frame, (center_x, center_y), 5, (255, 255, 255), 2)

        cv2.circle(red_frame, (320, 240), 5, (0, 255, 0), 2)
        cv2.circle(frame, (320, 240), 5, (0, 255, 0), 2)

        
        print(str(center_x) + " px", str(center_y) + " px", str(z) + " mm") # z is in mm

        # Z = depth value of the pixel (u,v) from the depth map
        # X = (( u - c_x) * Z) / (f_x)
        # Y = (( v - c_y) * Z) / (f_y)

        Z = z
        X = (( center_x - 320) * z) / (4300)
        Y = (( center_y - 240) * z) / (4300)
        print(str(round(X)) + " mm", str(round(Y)) + " mm", str(Z) + " mm") # z is in mm





        cv2.imshow("Frame", frame)
        cv2.imshow("red_frame", red_frame)
