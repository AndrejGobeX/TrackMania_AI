# Driver script
# s - begins drive
# f - finishes drive
# q - quits

from Engines import Neptune
import Mods
from PIL import ImageGrab
import keyboard
import DirectKey
import numpy as np
import time
import threading
import sys

w = 30
h = 30
t = 0.9

argv = str(sys.argv)
if len(sys.argv) == 2:
    w = int(argv[1])
elif len(sys.argv) == 3:
    w = int(argv[1])
    h = int(argv[2])
elif len(sys.argv) > 3:
    w = int(argv[1])
    h = int(argv[2])
    t = int(argv[3])

# image dimensions
image_width = w
image_height = h

treshold = t

# model = (your favorite engine goes here)
model = Neptune.Neptune(image_width, image_height)
model.load()

# mod = (desired image mod function)
mod = Mods.mod_neptune_crop

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