# Driver script
# s - begins drive
# f - finishes drive
# q - quits

from Engines.Neptune import Neptune
import Mods
from PIL import ImageGrab
import keyboard
import DirectKey
import numpy as np
import time
import threading
import sys

if len(sys.argv) != 2:
    exit()
model_name = sys.argv[1]

# model = (your favorite engine goes here)
model = globals()[model_name]()
model.load()

# image dimensions
image_width = model.image_width
image_height = model.image_height

treshold = 0.97

# mod = (desired image mod function)
mod = model.mod_function

up = False
down = False
left = False
right = False

KEY_UP = 0xC8
KEY_DOWN = 0xD0
KEY_LEFT = 0xCB
KEY_RIGHT = 0xCD

print("Ready")

while(not keyboard.is_pressed('q')):
    if not keyboard.is_pressed('s'):
        continue

    while(True):

        start = time.time()

        # screenshot
        img = ImageGrab.grab()

        img = mod(img, image_width, image_height)
        #print(img.shape)
        output = model.predict(np.array([img]))[0]
        output = output > treshold

        up = output[3]
        down = output[2]
        left = output[1]
        right = output[0]

        if up:
            DirectKey.PressKey(KEY_UP)
        else:
            DirectKey.ReleaseKey(KEY_UP)

        if down:
            DirectKey.PressKey(KEY_DOWN)
        else:
            DirectKey.ReleaseKey(KEY_DOWN)

        if left:
            DirectKey.PressKey(KEY_LEFT)
        else:
            DirectKey.ReleaseKey(KEY_LEFT)

        if right:
            DirectKey.PressKey(KEY_RIGHT)
        else:
            DirectKey.ReleaseKey(KEY_RIGHT)

        stop = time.time()
        print(stop - start) # reaction time

        if keyboard.is_pressed('f'):
            DirectKey.ReleaseKey(KEY_UP)
            DirectKey.ReleaseKey(KEY_DOWN)
            DirectKey.ReleaseKey(KEY_LEFT)
            DirectKey.ReleaseKey(KEY_RIGHT)
            break