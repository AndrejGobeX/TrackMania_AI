# Visualizer script
# q - quits
#
# python Visualizer.py width height

import cv2
from Screenshot import screenshot
import sys
import numpy as np


def perspective_warp(img):
    pts1 = np.float32([[0-(70*360//2),540//2+360//2],[1600//2+(70*360//2),540//2+360//2],[540//2,480//2],[1060//2,480//2]])
    pts2 = np.float32([[0,900//2],[1600//2,900//2],[0,0],[1600//2,0]])

    M = cv2.getPerspectiveTransform(pts1,pts2)
    dst = cv2.warpPerspective(img,M,(1600//2,900//2), borderMode=cv2.BORDER_CONSTANT, borderValue=[255, 255, 255])

    #dst = cv2.resize(dst, (width, height))
    #dst = cv2.medianBlur(dst, 5)
    #dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    #dst = (dst < 50) * np.uint8(255)

    return dst


def fun():
    return


if len(sys.argv) < 3:
    print('Not enough arguments.')
    exit()

width = int(sys.argv[1])
height = int(sys.argv[2])

lower_gray = np.array([0, 5, 50], np.uint8)
upper_gray = np.array([20, 5, 50], np.uint8)

while(True):

    frame = screenshot(w=width, h=height)
  
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()