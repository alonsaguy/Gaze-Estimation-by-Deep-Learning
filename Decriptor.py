import matplotlib.pyplot as plt
import numpy as np

def convert_angle_to_screen(current_gaze):
    # Assumptions: ScreenSize = [34.5,19.5] [cm], eyeLocation [cm], DistOfCamFromTheScreen = -2 [cm]
    # current_gaze provides (pitch, yaw) angles
    ScreenSize = [34.5,19.5] # cm units (Alon's Computer)
    DistOfCamFromTheScreen = 2.0 # cm units
    eyeLocation = [0.0, 5.0, 40.0] # cm units (relative to the camera)
    
    lookVec = [eyeLocation[2]*np.tan(current_gaze[1]),eyeLocation[2]*np.tan(current_gaze[0]), -eyeLocation[2]]
    T = eyeLocation[2]/float(lookVec[2])
    
    # The sign is + because the camera flips the look vector in x axis
    return [eyeLocation[0] + T*float(lookVec[0]) + ScreenSize[0]/2, eyeLocation[1] - T*float(lookVec[1]) - DistOfCamFromTheScreen]
