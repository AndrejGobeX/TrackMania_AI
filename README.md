# TrackMania_AI
Neural network driver for Trackmania
## Intro and goals
Computer vision and self-driving cars are one of the most common topics in ML/AI nowadays. As I am new in the ML/AI world, I wanted to experiment and play with AI cars.
I played Trackmania when I was young and figured that it could be a good environment for this project.\
Goal here is to make a decent driver (read: safe driver). It is not (yet) a racing material, but in the future I might try to make something faster.\
If you have any ideas, comments or improvements, let me know.
## Details
It is currently a supervised framework (meaning it learns from the dataset, not on it's own). I plan to do reinforcement/evolution/mutation net in the future (if anyone know the easiest way to link NEAT or Tensorflow's reinforcement API with Trackmania, let me know).\
I've split the dataset into two categories: third_person and first_person, based on the camera.\
The main problem is the processing time while driving. It takes approx. 0.3s from one shot to another on my machine which is very bad. At full throttle, it is impossible not to hit a wall.\
I tried to add speed limit which does make a small improvement, but it's a hack, so i'm trying to avoid it.
Speed capturing is done with [CheatEngine](https://www.cheatengine.org/) and ctypes in Python. It's a little tricky to set it all up because speed variable shifts it's location on restart.\
Networks are located in Engines/ directory.
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
