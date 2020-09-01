import numpy as np

# This module calculates the pitch and yaw angles of the camera relative to the eye and 
# the suitable pitch and yaw ranges for the gaze vector of the eye.
# INPUT: ScreenSize [cm], DistOfCamFromScreen [cm], eyeInCamSpace [cm] which refer to: the actual screen size
# the actual place of the camera from the edges of the screen and the eye location in camera space (the camera is
# located in the origin)
# We use this information to choose what parameters we should give as inputs to UnityEyes tool.

# Inputs
# TODO: the inputs are currently arbitrary
ScreenSize = [34.5,19.5] # cm units (Alon's Computer)
DistOfCamFromTheScreen = 2.0 # cm units
eyeInCamSpace = [5.0, 15.0 + DistOfCamFromTheScreen,60.0] # cm units (relative to the camera)

# Calculations
Dist = np.sqrt(eyeInCamSpace[0]**2+eyeInCamSpace[1]**2+eyeInCamSpace[2]**2)
PitchAngle_Cam = -np.arcsin(eyeInCamSpace[1]/Dist)*180/np.pi #First param for the camera
YawAngle_Cam = -np.arcsin(eyeInCamSpace[0]/Dist)*180/np.pi #Second param for the camera

PitchRange_Eye = np.arctan((ScreenSize[1]/2)/eyeInCamSpace[2])*180/np.pi #Third param for the eye
YawRange_Eye = np.arctan((ScreenSize[0]/2)/eyeInCamSpace[2])*180/np.pi #Fourth param for the eye

# Outputs
print("The camera parameters should be: (",PitchAngle_Cam,", 0 ,",YawAngle_Cam,", 0 )")
print("The eye parameters should be: ( ",-PitchAngle_Cam , " , 0 ",PitchRange_Eye,",",YawRange_Eye, ")")