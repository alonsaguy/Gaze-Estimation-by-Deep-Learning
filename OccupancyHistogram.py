import json
import os.path
from builtins import round
import matplotlib.pyplot as plt

# Assumptions: ScreenSize = [34.5,19.5] cm, eyeLocation = [0,11.75,60] cm, DistOfCamFromTheScreen = -2 cm
ScreenSize = [34.5, 19.5]  # cm units (Alon's Computer)
DistOfCamFromTheScreen = 2.0  # cm units
eyeLocation = [0.0, ScreenSize[1] / 2 + DistOfCamFromTheScreen, 60.0]  # cm units (relative to the camera)

histogramX = []
histogramY = []

# path joining version for other paths
DIR = os.path.join(os.getcwd(), r'UnityEyes/imgs/')
DataSetSize = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])

for i in range(1, round(DataSetSize / 2)):
    with open(r'UnityEyes/imgs/{0}.json'.format(i)) as jsonFile:
        data = json.load(jsonFile)
        eyeDetails = data['eye_details']
        Vec = eyeDetails['look_vec']
        lookVec = Vec[1:-1].split(',')
        T = eyeLocation[2] / float(lookVec[2])

        Xscreen = eyeLocation[0] + T * float(lookVec[0]) + ScreenSize[0] / 2  # The sign is + because the camera flips the look vector in x axis
        Yscreen = eyeLocation[1] - T * float(lookVec[1]) - DistOfCamFromTheScreen
        histogramX.append(Xscreen)
        histogramY.append(Yscreen)


bins=100
plt.hexbin(histogramX, histogramY, bins=bins, gridsize=1000)   # <- You need to do the hexbin plot
plt.plot([0,0,ScreenSize[0],ScreenSize[0],0],[0,ScreenSize[1],ScreenSize[1],0,0],color="red")
cb = plt.colorbar()
cb.set_label('density')
plt.xlim([0, 34.5])
plt.ylim([0, 19.5])
plt.show()