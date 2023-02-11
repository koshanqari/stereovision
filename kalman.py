#code fully explained below
import numpy as np
from pykalman import KalmanFilter

# Define the initial state and transition matrix
# Assuming a constant velocity model
initial_state_mean = [0, 0, 0, 0, 0, 0]
transition_matrix = [[1, 0, 0, 1, 0, 0],
                     [0, 1, 0, 0, 1, 0],
                     [0, 0, 1, 0, 0, 1],
                     [0, 0, 0, 1, 0, 0],
                     [0, 0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 0, 1]]

# Define the observation matrix and observation noise covariance matrix
observation_matrix = [[1, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0]]

observation_covariance = np.eye(3) * 0.1

# Define the process noise covariance matrix
process_covariance = np.eye(6) * 0.001

# Create the Kalman Filter object
kf = KalmanFilter(transition_matrices=transition_matrix,
                  observation_matrices=observation_matrix,
                  initial_state_mean=initial_state_mean,
                  initial_state_covariance=np.eye(6),
                  observation_covariance=observation_covariance,
                  process_covariance=process_covariance)

# Initialize the state estimate
state_estimate = initial_state_mean

while True: # Continuously update the state estimate as new data arrives
    # Get the latest x, y, z observation
    x, y, z = get_new_observation()
    observation = np.array([x, y, z])

    # Update the state estimate
    (state_estimate, state_covariance) = kf.filter_update(state_estimate, state_covariance, observation)
    
    # Use the state estimate for your purposes
    ...



# -------------------------------------------------------------------EXPLINATION --------------------------------------------------------------

# The PyKalman library is a library that provides an implementation of the Kalman Filter, 
# which is a popular algorithm for estimating the state of a dynamic system when observations of
#  that system are made over time. In the context of x, y, z coordinates, 
# you can use PyKalman to filter out the noise in a series of such observations to 
# produce a more accurate estimate of the true position of the system over time.

import numpy as np
from pykalman import KalmanFilter

# Define the initial state and transition matrix
# Assuming a constant velocity model
initial_state_mean = [0, 0, 0, 0, 0, 0]  #-->   his is basically the initial [x, y, z, vx, vy, vz]
transition_matrix = [[1, 0, 0, 1, 0, 0], # -->  In the context of a Kalman filter, the transition 
                     [0, 1, 0, 0, 1, 0], #      matrix is a matrix that defines the relationship between the 
                     [0, 0, 1, 0, 0, 1], #      state of the system at the current time step and the state of the system at the next time step.
                     [0, 0, 0, 1, 0, 0],
                     [0, 0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 0, 1]]

# Define the observation matrix and observation noise covariance matrix
observation_matrix = [[1, 0, 0, 0, 0, 0], #--->the observation matrix defines the mapping between the 
                      [0, 1, 0, 0, 0, 0], #    true state of the system and the measurements that are made.
                      [0, 0, 1, 0, 0, 0]]

observation_covariance = np.eye(3) * 0.1 #The observation covariance matrix is used in the Kalman filter to 
                                         #weight the importance of the measurements in correcting 
                                         # the state estimate. A larger value of the observation covariance indicates that 
                                         # the measurements are less reliable, and the Kalman filter will give less weight 
                                         # to these measurements in correcting the state estimate. Conversely, 
                                         # a smaller value of the observation covariance indicates that the measurements 
                                         # are more reliable, and the Kalman filter will give more weight to these 
                                         # measurements in correcting the state estimate.

# Define the process noise covariance matrix
process_covariance = np.eye(6) * 0.001  #The process covariance matrix is used in the Kalman filter 
                                        #to model the process noise, which represents the unpredictable 
                                        # variations in the state of the system over time. 
                                        # A larger value of the process covariance indicates that the 
                                        # process noise is larger and that the state of the system is 
                                        # more uncertain, whereas a smaller value of the process covariance 
                                        # indicates that the process noise is smaller and that the state of 
                                        # the system is more predictable.

# Create the Kalman Filter object
kf = KalmanFilter(transition_matrices=transition_matrix,
                  observation_matrices=observation_matrix,
                  initial_state_mean=initial_state_mean,
                  initial_state_covariance=np.eye(6),
                  observation_covariance=observation_covariance,
                  process_covariance=process_covariance)

# # Provide the observations
# observations = np.array([[x, y, z] for x, y, z in zip(x_coordinates, y_coordinates, z_coordinates)])

# # Filter the observations
# (filtered_state_means, filtered_state_covariances) = kf.filter(observations)


# Initialize the state estimate
state_estimate = initial_state_mean

while True: # Continuously update the state estimate as new data arrives
    # Get the latest x, y, z observation
    x, y, z = get_new_observation()
    observation = np.array([x, y, z])

    # Update the state estimate
    (state_estimate, state_covariance) = kf.filter_update(state_estimate, state_covariance, observation)
    
    # Use the state estimate for your purposes
    ...
