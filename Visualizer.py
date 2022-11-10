# Visualizer script
# q - quits
#
# python Visualizer.py width height

import cv2
from Screenshot import screenshot
import sys
import numpy as np
from time import time


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

def color_quantization(img, K):
    Z = img.reshape((-1,3))
    
    # convert to np.float32
    Z = np.float32(Z)
    
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret,label,center=cv2.kmeans(Z, K, None, criteria, 20, cv2.KMEANS_RANDOM_CENTERS)
    
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    return res.reshape((img.shape))

from Window import WindowInterface

#window = WindowInterface()
#window.move_and_resize()

lower = np.array([0,0,0], dtype = "uint8")
upper = np.array([180,255,40], dtype = "uint8")

#cv2.imwrite("prev.png", window.screenshot())
img = cv2.imread("prev.png")

while(True):

    #frame = screenshot()
    #frame = window.screenshot()
    #frame = cv2.cvtColor(window.screenshot(), cv2.COLOR_BGRA2BGR)
    frame = img.copy()
    
    start = time()
    frame = frame[225:][:-40]
    frame = cv2.resize(frame, (200, 100))
    frame = cv2.Canny(frame, 100, 50)
    #frame[:,:,0]=0
    #frame[:,:,1]=0
    #frame = color_quantization(frame, 4)
    print(time() - start)

    #mask = cv2.inRange(frame, lower, upper)
    #frame = cv2.Canny(mask, 100, 150)

    #print(frame.shape)
    
    
    #frame = cv2.Canny(frame, 100, 200)
  
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()