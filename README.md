# TrackMania_AI
Neural network driver for Trackmania
## Intro and goals
Computer vision and self-driving cars are one of the most common topics in ML/AI nowadays. As I am new in the ML/AI world, I wanted to experiment and play with AI cars.
I played Trackmania when I was young and figured that it could be a good environment for this project.\
Goal here is to make a decent driver. It is not (yet) a racing material.\
If you have any ideas, comments or improvements, let me know.
## Details
This project uses both supervised and reinforcement approaches (RL branch is dedicated to the latter one).\
I've split the dataset into two categories: third_person and first_person, based on the camera.\
Speed capturing is now done with [Openplanet](https://openplanet.nl/) and sockets in Python.
Networks are located in Engines/ directory.
## Current networks
For now, I've made two supervised networks and one neuroevultion configuration. Each of them outputs a 4-number array which represents probabilities of certain arrow keys being pressed (0.0 - 1.0). Threshold for each of the arrows can be tweaked independetly. In the future, it will be tied to the model.\
Neptune is a convolutional network containing two conv2d layers. It just takes a 50x50 modded image and pops out four probabilities.\
![Neptune mod](https://github.com/AndrejGobeX/TrackMania_AI/blob/main/Engines/neptune_mod.png?raw=true)\
Maniack, on the other hand, uses a 10-number vector as input. It represents normalized distances from the bottom of the modded image to the nearest black pixel above. This can be enhanced by using more lines and using horizontal distances measured from the middle, which I might try next time.\
![Maniack mod](https://github.com/AndrejGobeX/TrackMania_AI/blob/main/Engines/maniack_mod.png?raw=true)
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
## Setup
I have tested this procedure and it should work on any computer. The main problems here are resolution and camera. You should edit the scripts with respect to your resolution and, while in Trackmania, switch to first person (currently, nets are trained with first person because it is a lot easier for AI to understand movement and to follow the track that way).
1. Download Trackmania and Python with all the necessary packages.
2. Download Openplanet and install it.
3. Git clone this repository.
3. Copy the `Plugin*` files from `Scripts\` directory to `C:\Users\(username)\OpenplanetNext\Scripts`.
4. Run Trackmania, press F3 to open Openplanet, go to Openplanet > Log and check if the script is running (should print Waiting for incomming connection). You can, if necessary, reload the plugin from Developer > Reload plugin. Open any map (nets are trained on basic circuits, so the best thing is to create a new one), and enter driving mode.
5. Alt-tab or press Win key and open command prompt. Find Driver.py and enter `python Driver.py Neptune` (or Maniack).
6. Wait until you see Press s to begin, alt-tab to Trackmania, unpause the game and press `S`.
7. The car should go autonomously. To end a run, press `F`, to quit the script, press `Q`.
## Packages used and credits
I would like to thank Yann and Edouard (@trackmania-rl) for making the script for openplanet. I have included the licence and copyright notice with those files in the `Scripts\` directiory.
Big thanks to the community!\
The latest versions are used for all packages (31.10.2021.):
* python-pillow/Pillow - image preprocessing
* boppreh/keyboard - keyboard API
* numpy/numpy - array manipulation
* opencv/opencv - image preprocessing
* tensorflow - ML API
