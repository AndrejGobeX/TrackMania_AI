from PIL import ImageGrab
import numpy as np
import keyboard
import pathlib

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

# result = cv2.VideoWriter('video_'+str(log)+'.avi', cv2.VideoWriter_fourcc(*'MJPG'), 60, size) 
x=0

pathlib.Path('./images/0000').mkdir(parents=True, exist_ok=True)
pathlib.Path('./images/0001').mkdir(parents=True, exist_ok=True)
pathlib.Path('./images/0010').mkdir(parents=True, exist_ok=True)
pathlib.Path('./images/0011').mkdir(parents=True, exist_ok=True)
pathlib.Path('./images/0100').mkdir(parents=True, exist_ok=True)
pathlib.Path('./images/0101').mkdir(parents=True, exist_ok=True)
pathlib.Path('./images/0110').mkdir(parents=True, exist_ok=True)
pathlib.Path('./images/0111').mkdir(parents=True, exist_ok=True)
pathlib.Path('./images/1000').mkdir(parents=True, exist_ok=True)
pathlib.Path('./images/1001').mkdir(parents=True, exist_ok=True)
pathlib.Path('./images/1010').mkdir(parents=True, exist_ok=True)
pathlib.Path('./images/1011').mkdir(parents=True, exist_ok=True)
pathlib.Path('./images/1100').mkdir(parents=True, exist_ok=True)
pathlib.Path('./images/1101').mkdir(parents=True, exist_ok=True)
pathlib.Path('./images/1110').mkdir(parents=True, exist_ok=True)
pathlib.Path('./images/1111').mkdir(parents=True, exist_ok=True)

keyboard.wait('s')

while(True):
    x=x+1

    up = int(keyboard.is_pressed('up'))
    down = int(keyboard.is_pressed('down'))
    left = int(keyboard.is_pressed('left'))
    right = int(keyboard.is_pressed('right'))

    image = './images/' + str(up+down*10+left*100+right*1000+10000)[1:] + '/' + str(log) + '_' + str(x) + '.jpg'
	
    frame = ImageGrab.grab()
    frame.save(image)
    print(image)

    # frame = np.array(frame) 
    # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # result.write(frame) 
    # cv2.imshow('Frame', frame) 
    if(keyboard.is_pressed('f')):
        break

# result.release() 
# cv2.destroyAllWindows() 

file = open(logfile, 'w')
file.write(str(log))
file.close()