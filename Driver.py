# Driver script
# s - begins drive
# f - finishes drive
# q - quits
#
# python Driver.py Neptune 7070 0xABCDEFG

import os
from PIL import ImageGrab
import keyboard
import numpy as np
import time
import sys
import SpeedCapture
import DirectKey

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# first parameter is the name of the engine class

if len(sys.argv) < 4:
    print('Not enough arguments.')
    exit()

# get the engine class

model_name = sys.argv[1]
engine_module = __import__('Engines.' + model_name, fromlist=[model_name])
engine_class = getattr(engine_module, model_name)

model = engine_class()
model.load()

# trackmania PID, speed address and endian

PID = int(sys.argv[2])
address = sys.argv[3]
if address[:2] != '0x':
    address = '0x' + address
address = int(address, 0)

if len(sys.argv) >= 4:
    endian = 'little'
else:
    endian = sys.argv[4]

# image dimensions
image_width = model.image_width
image_height = model.image_height

threshold = 0.9999

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

while not keyboard.is_pressed('q'):
    if not keyboard.is_pressed('s'):
        continue

    while True:

        # uncomment for reaction time measurement
        # start = time.time()

        # screenshot
        img = ImageGrab.grab()

        img = mod(img, model)

        try:
            img = img / 255.0
        except:
            img = img

        # speed
        speed = SpeedCapture.GetSpeed(PID, address, endian=endian)

        x = (np.array([img]), np.array([speed]))

        output = model.predict(x)[0]
        output = output > threshold

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

        # stop = time.time()
        # print(stop - start) # reaction time

        if keyboard.is_pressed('f'):
            DirectKey.ReleaseKey(KEY_UP)
            DirectKey.ReleaseKey(KEY_DOWN)
            DirectKey.ReleaseKey(KEY_LEFT)
            DirectKey.ReleaseKey(KEY_RIGHT)
            break
