# Gaze-Estimation-by-Deep-Learning

This project goal is to track a person's focusing coordinate on a monitor.
The algorithm estimated the gaze direction using convolutional neural network (CNN).

The training set is using simulated images of human eyes created by eye simulator named UnityEyes [1].
The eyes simulator is available at: https://www.cl.cam.ac.uk/research/rainbow/projects/unityeyes/

![GazeEstimation](/GazeEstimation.png)

**Code structure:**
* main.py - calls for training and testing routines  
* core.py - controls the neural network runs
* elg.py - containing eye lendmarks model and estimation methods
* UnityEyes.py - containing function that reads UnityEyes data sets  
* util.py - usefull functions for the entire project
* video.py - manages the display on the monitor
* decriptor.py - tranlates the gaze vector to a coordinate on the monitor
* OccupancyHistogram.py - checks simulated data quality
* ScreenFit.py - a basic example for calibration of the final estimation to one person
  

The train set images should be in the following folder: UnityEyes\imgs\
On the train phase the model parameters would be saved in a directory named "outputs".
On the test phase you might control the parameter show_vid in the main.py file to either option:
* True - showing a real time video of the person with the estimated gaze vector
* False - showing the estimated focus coordinates on the screen

Our project is based on previous work by Seonwook Park, et al in https://github.com/swook/GazeML [2], [3]

**Refrences:**

[1] Wood E, et al. “Learning an Appearance-Based Gaze Estimator from One Million Synthesized
Images”. ETRA, 131-138 (2016).

[2] Seonwook Park, Xucong Zhang, Andreas Bulling, and Otmar Hilliges. "Learning to find eye region landmarks for remote gaze estimation in unconstrained settings." In Proceedings of the 2018 ACM Symposium on Eye Tracking Research & Applications, p. 21. ACM, (2018).

[3] Seonwook Park, Adrian Spurr, and Otmar Hilliges. "Deep Pictorial Gaze Estimation". In European Conference on Computer Vision (2018).
