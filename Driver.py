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

# image dimensions
image_width = 100
image_height = 100

treshold = 0.3

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

        if keyboard.is_pressed('f'):
            DirectKey.ReleaseKey(KEY_UP)
            DirectKey.ReleaseKey(KEY_DOWN)
            DirectKey.ReleaseKey(KEY_LEFT)
            DirectKey.ReleaseKey(KEY_RIGHT)
            break