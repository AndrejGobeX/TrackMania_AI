from PIL import ImageGrab, Image
import numpy as np
import keyboard
import pathlib, shutil
import sys

args = str(sys.argv)
if len(sys.argv) == 1:
    driver_camera = 'third_person'
else:
    driver_camera = 'first_person'

logfile = './log.txt'

if pathlib.Path(logfile).exists():
    file = open(logfile, 'r')
    log = int(file.read()) + 1
    file.close()
else:
    log = 0


img = ImageGrab.grab()

frame_width = img.width 
frame_height = img.height
print(frame_width)
print(frame_height) 

size = (frame_width, frame_height) 

x=0

image_dir = './images/' + driver_camera + '/'
temp_dir = './temp'

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
pathlib.Path(temp_dir).mkdir(parents=True, exist_ok=True)

while(not keyboard.is_pressed('q')):
    current_run = []
    if not keyboard.is_pressed('s'):
        continue

    while(True):
        x=x+1

        up = int(keyboard.is_pressed('up'))
        down = int(keyboard.is_pressed('down'))
        left = int(keyboard.is_pressed('left'))
        right = int(keyboard.is_pressed('right'))

        temp_image = temp_dir + '/' + str(log) + '_' + str(x) + '.jpg'
        image = image_dir + str(up+down*10+left*100+right*1000+10000)[1:] + '/' + str(log) + '_' + str(x) + '.jpg'
        
        frame = ImageGrab.grab()
        frame.save(temp_image)
        print(temp_image)

        current_run.append([temp_image, image])

        if(keyboard.is_pressed('x')):
            break

        if(keyboard.is_pressed('f')):
            for shot in current_run:
                temp_image = shot[0]
                image = shot[1]

                Image.open(temp_image).save(image)
                # print(image)
            break
            print(len(shot))
    
    shutil.rmtree(pathlib.Path(temp_dir))
    pathlib.Path(temp_dir).mkdir(parents=True, exist_ok=True)

file = open(logfile, 'w')
file.write(str(log))
file.close()