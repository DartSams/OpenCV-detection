# OpenCV-detection

![MediaPipe hand map](hand_landmarks_map.png)


To make your own haarcascades
- use cascade trainer GUI (https://amin-ahmadi.com/cascade-trainer-gui/)

Step 1.
Open the cascade trainer GUI and select the image folder you want to use for training.

Step 2.
In your image folder make two sub folder n and p for negative images and positive images. The positives are what you want to neural network to identify and the negatives are what you want to train against. Remember the more pictures the better.

Step 3.
Select the type haar are easy to use and the default but hog are more powerful and accurate.

Step 4.
Click start to train.

Step 5.
After training is finished in the image folder it will have created a classifier folder in that is the cascade.xml this file is what is useful for identifying objects in opencv.

