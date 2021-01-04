# Records gameplay
# s - starts recording
# f - finishes recording
# x - same as f, but cancels (no shots will be saved)
# q - quits recorder
# python ScreenRecorder.py third_person PID address little

from PIL import ImageGrab
import numpy as np
import keyboard
import pathlib
import sys

import SpeedCapture

if len(sys.argv) < 4:
    print('Not enough arguments.')
    exit()

driver_camera = sys.argv[1]

# trackmania PID, speed address and endian

PID = int(sys.argv[2])
address = int(sys.argv[3], 0)

if len(sys.argv) == 4:
    endian = 'little'
else:
    endian = sys.argv[4]

# old recorder

logfile = './log.txt'

if pathlib.Path(logfile).exists():
    file = open(logfile, 'r')
    log = int(file.read()) + 1
    file.close()
else:
    log = 0


img = ImageGrab.grab()

# rldu

frame_width = img.width 
frame_height = img.height
print(frame_width)
print(frame_height) 

size = (frame_width, frame_height) 

x=0

image_dir = './images/' + driver_camera + '/'

pathlib.Path(image_dir + '0000').mkdir(parents=True, exist_ok=True)
pathlib.Path(image_dir + '0001').mkdir(parents=True, exist_ok=True)
pathlib.Path(image_dir + '0010').mkdir(parents=True, exist_ok=True)
pathlib.Path(image_dir + '0011').mkdir(parents=True, exist_ok=True)
pathlib.Path(image_dir + '0100').mkdir(parents=True, exist_ok=True)
pathlib.Path(image_dir + '0101').mkdir(parents=True, exist_ok=True)
pathlib.Path(image_dir + '0110').mkdir(parents=True, exist_ok=True)
pathlib.Path(image_dir + '0111').mkdir(parents=True, exist_ok=True)
pathlib.Path(image_dir + '1000').mkdir(parents=True, exist_ok=True)
pathlib.Path(image_dir + '1001').mkdir(parents=True, exist_ok=True)
pathlib.Path(image_dir + '1010').mkdir(parents=True, exist_ok=True)
pathlib.Path(image_dir + '1011').mkdir(parents=True, exist_ok=True)
pathlib.Path(image_dir + '1100').mkdir(parents=True, exist_ok=True)
pathlib.Path(image_dir + '1101').mkdir(parents=True, exist_ok=True)
pathlib.Path(image_dir + '1110').mkdir(parents=True, exist_ok=True)
pathlib.Path(image_dir + '1111').mkdir(parents=True, exist_ok=True)

clear_list = []

while(not keyboard.is_pressed('q')):
    if not keyboard.is_pressed('s'):
        continue

    temp_stint = []
    while(True):
        x=x+1

        frame = ImageGrab.grab()
        
        speed = SpeedCapture.GetSpeed(PID, address, endian=endian)

        up = int(keyboard.is_pressed('up'))
        down = int(keyboard.is_pressed('down'))
        left = int(keyboard.is_pressed('left'))
        right = int(keyboard.is_pressed('right'))

        image = image_dir + str(up+down*10+left*100+right*1000+10000)[1:] + '/' + \
            str(log) + '_' + str(x) + '_' + str(speed) + '.jpg'
        
        temp_stint.append(image)
        
        frame.save(image)
        print(image)

        if(keyboard.is_pressed('x')):
            clear_list.append(temp_stint)
            break

        if(keyboard.is_pressed('f')):
            break

for stint in clear_list:
    for image in stint:
        pathlib.Path(image).unlink()

file = open(logfile, 'w')
file.write(str(log))
file.close()