# TrackMania_AI
Neural network driver for Trackmania
## Intro and goals
Computer vision and self-driving cars are one of the most common topics in ML/AI nowadays. As I am new in the ML/AI world, I wanted to experiment and play with AI cars.
I played Trackmania when I was young and figured that it could be a good environment for this project.\
New goal is to make a competitive driver (against other AI's) and, hopefully, advance this project to the real car (we have already tested a supervised net on an RC car, gif below).\
If you have any ideas, comments or improvements, let me know.

<p align="center">
     <video width="320" height="240" controls>
        <source src="demo2.mp4" type="video/mp4">
    </video> 
</p>
<p align="center">
    <i>AI in action</i>
</p>

<p align="center">
    <img src="rc_car_preview.gif" alt="animated" width="500"/>
</p>
<p align="center">
    <i>Toy car demo</i>
</p>


## Details
**This branch deals with a new gymnasium env for reinforcement algorithms. Please view the old `main` branch for supervised, neuroevuolution and old reinforcement algorithms.**\
The `TrackmaniaEnv.py` file implements the new env for RL. You can tweak the number of previous observations and actions that will be stacked into the next observation.
Lately, I am working on implementing the SAC algorithm, as it has shown great results in various enviromnents. I have made a gym that implements a simplified version of the reward function presented in a paper by Florian Fuchs, Yunlong Song, Elia Kaufmann, Davide Scaramuzza, and Peter Dürr ([Super-Human Performance in Gran Turismo Sport Using Deep Reinforcement Learning](https://rpg.ifi.uzh.ch/docs/RAL21_Fuchs.pdf)). In the video([link](https://youtu.be/Zeyv1bN9v4A)) it can be seen how good the AI performs. As for this repo, it is still in development. I would also like to try out different algorithms with the same gym in order to benchmark the results.
## Contents
Since I am focusing more on the SAC right now, some scripts are left unchanged. You are free to edit them if you find something messy. Files with "*" are discontinued for now.
| Filename | Brief description |
| -------- | ----------------- |
| MapExtractor/Program.cs | C# code that extracts track edges from a .gbx map file |
| Commands.py | Wrappers for game keys and virtual gamepad |
| Dataset.py* | Contains functions which take a dataset and preprocess it |
| DirectKey.py | Contains ctypes functions for key presses ( *pip packages won't work* ) |
| Driver.py* | Runs in the background while the game is running and presses buttons |
| GetData.py | Functions for getting in game data (speed, distance, etc.) |
| MapExtractor.py | Converts .txt files processed by C# script into an array |
| Mods.py* | Functions for data preprocessing |
| NEAT_Trainer.py* | Neuroevolution trainer |
| RL_Driver.py | Driver for the SAC |
| ScreenRecorder.py* | Background script to capture frames while you play |
| Screenshot.py* | Fast screenshot functions |
| SpeedCapture.py* | ~~Functions to get speed value from the game~~ (*deprecated*) |
| TMEnv.py | Gym environment for SAC |
| Trainer.py* | Trains the neural network |
| Trainer.ipynb* | Starting point for Colab training (if you do not wish to train on your computer) |
| Visualizer.py* | Visualizes a modded screenshot while driving |
## Setup
I expect you have basic knowledge of python. Explore python and openplanet and check the issues before making a new one.\
I have tested this procedure and it should work on any computer. ~~I am working on windowed mode, but for now it only works in fullscreen mode.~~ **Supervised and NEAT algorithms are for now not updated. I will put a simple tutorial on how to run them, but you can edit them freely if something does not work. These algorithms work only in fullscreen, however I have uploaded a screenshoting method that uses windowed mode that you can implement.** The main problem here is the camera, so you should switch to first person (press `3` twice to go full first person, screenshots are above) (currently, nets are trained with first person because it is a lot easier for AI to understand movement and to follow the track that way). `RL_Driver.py` and SAC do not use screenshots, so you can resize the screen freely.\
**YOU DO NOT NEED STANDARD/CLUB ACCESS TO USE OPENPLANET AND THIS PROJECT.**\
As mentioned in the issues, Openplanet won't load the plugin if run from EpicGames launcher. If you encounter these problems, try running Trackmania from .exe or use Ubisoft Connect.
1. Download Trackmania and Python with all the necessary packages (package info below).
2. Download Openplanet and install it.
3. Git clone this repository.
3. Copy the `Plugin*` (version 0.2 has added direction support) files from `Scripts\` directory to `C:\Users\(username)\OpenplanetNext\Scripts`. (If there is no `Scripts` directory in `OpenplanetNext`, run Trackmania and exit it)
4. Run Trackmania, press `F3` to open Openplanet, go to Openplanet > Log and check if the script is running (should print Waiting for incomming connection). You can, if necessary, reload the plugin from Developer > Reload plugin. Open any map (nets are trained on basic circuits, so the best thing is to create a new one), and enter driving mode. Press `F3` again to close the openplanet overlay (scripts capture full-screen and overlays will mess them up).
5. Alt-tab or press Win key and open command prompt.
    * If using supervised nets:
        1. Find Driver.py and enter `python Driver.py Neptune` (or Maniack).
        2. Wait until you see Press s to begin, alt-tab to Trackmania, unpause the game and press `S`.
        3. The car should go autonomously. To end a run, press `F`, to quit the script, press `Q`.
    * If using neuroevolution, check the NEAT_Trainer.py
    * If using SAC, run `python RL_Driver.py .\Maps\Mini.Map.txt` (or choose a different preprocessed map file). Note that you should use the map file of the track you are currently on.
## Credits
Big thanks to:
* Yann and Edouard (@trackmania-rl) for making the script for openplanet. I have included the licence and copyright notice with those files in the `Scripts\` directiory.
* Petr (@BigBang1112) for making a library that is used to extract track data from a map file.
* Antonin, Ashley, Adam, Anssi, Maximilian and Noah (@DLR-RM) for making Stable-Baselines3 used to apply reinforcement learning algorithms with ease.
* Everyone from Openplanet (@openplanet-nl) for creating the API used to extract in-game data from Trackmania.
* Alan, Matt, Cesar, Carolina and Marcio Lobo (@CodeReclaimers) for making neat-python used for neuroevolution.

If I missed quoting someone, please open an issue and I will update this page.
## Packages
The pip freeze output of the packages:
* gym==0.22.0
* keyboard==0.13.5
* matplotlib==3.4.3
* neat-python==0.92
* numpy==1.19.5
* opencv-python==4.5.4.58
* Pillow==8.4.0
* pywin32==303
* rtgym==0.6
* stable-baselines3==1.4.0
* tensorboard==2.6.0
* tensorboard-data-server==0.6.1
* tensorboard-plugin-wit==1.8.0
* tensorflow==2.5.0
* tensorflow-estimator==2.5.0
* torch==1.10.2
* vgamepad==0.0.5