# TrackMania_AI
Neural network driver for Trackmania
## Intro and goals
Computer vision and self-driving cars are one of the most common topics in ML/AI nowadays. As I am new in the ML/AI world, I wanted to experiment and play with AI cars.
I played Trackmania when I was young and figured that it could be a good environment for this project.\
Goal here is to make a decent driver. It is not (yet) a racing material.\
If you have any ideas, comments or improvements, let me know.
## Details
It is currently a supervised framework (meaning it learns from the dataset, not on it's own). I plan to do reinforcement/evolution/mutation net in the future (if anyone knows the easiest way to link NEAT or Tensorflow's reinforcement API with Trackmania, let me know).\
I've split the dataset into two categories: third_person and first_person, based on the camera.\
Speed capturing is done with [CheatEngine](https://www.cheatengine.org/) and ctypes in Python. It's a little tricky to set it all up because speed variable shifts it's location on restart.\
Networks are located in Engines/ directory.
## Current networks
For now, I've made two networks which use different approaches. Each of them outputs a 4-number array which represents probabilities of certain arrow keys being pressed (0.0 - 1.0). Threshold for each of the arrows can be tweaked independetly. In the future, it will be tied to the model.\ \
Neptune is a convolutional network containing two conv2d layers. It just takes a 50x50 modded image and pops out four probabilities.\
![Neptune mod](https://github.com/AndrejGobeX/TrackMania_AI/blob/main/Engines/neptune_mod.png?raw=true)\
Maniack, on the other hand, uses a 10-number vector as input. It represents normalized distances from the bottom of the modded image to the nearest black pixel above. This can be enhanced by using more lines and using horizontal distances measured from the middle, which I might try next time.\
![Maniack mod](https://github.com/AndrejGobeX/TrackMania_AI/blob/main/Engines/Maniack_mod.png?raw=true)
## Contents
| Filename | Brief description |
| -------- | ----------------- |
| Dataset.py | Contains functions which take a dataset and preprocess it |
| DirectKey.py | Contains ctypes functions for key presses ( *pip packages won't work* ) |
| Driver.py | Runs in the background while the game is running and presses buttons |
| Mods.py | Functions for data preprocessing |
| ScreenRecorder.py | Background script to capture frames while you play |
| SpeedCapture | Functions to get speed value from the game |
| Trainer.py | Trains the neural network |
## Packages used
Big thanks to the community!
* python-pillow/Pillow - image preprocessing
* boppreh/keyboard - keyboard API
* numpy/numpy - array manipulation
* tensorflow - ML API
