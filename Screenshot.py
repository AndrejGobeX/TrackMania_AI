import numpy as np
from time import time, sleep
import win32gui
import win32ui
import win32con
import cv2
import keyboard


def screenshot(w=800, h=450):
    
    hwnd = win32gui.FindWindow(None, "Trackmania")
    x, y, x1, y1 = win32gui.GetWindowRect(hwnd)
    if x1-x == w:
        #fullscreen
        borders = (0, 0)
    else:
        #window
        borders = (8, 31)
    wDC = win32gui.GetWindowDC(hwnd)
    dcObj=win32ui.CreateDCFromHandle(wDC)
    cDC=dcObj.CreateCompatibleDC()
    dataBitMap = win32ui.CreateBitmap()
    dataBitMap.CreateCompatibleBitmap(dcObj, w, h)
    cDC.SelectObject(dataBitMap)
    cDC.BitBlt((0,0),(w, h) , dcObj, borders, win32con.SRCCOPY)
    img_array = dataBitMap.GetBitmapBits(True)
    img = np.frombuffer(img_array, dtype='uint8')
    img.shape = (h, w, 4)
    dcObj.DeleteDC()
    cDC.DeleteDC()
    win32gui.ReleaseDC(hwnd, wDC)
    win32gui.DeleteObject(dataBitMap.GetHandle())

    return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)


def screenshot2(w=800, h=450):
    
    hwnd = win32gui.FindWindow(None, "Trackmania")
    #wDC = win32gui.GetWindowDC(hwnd)
    x, y, x1, y1 = win32gui.GetClientRect(hwnd)
    x, y = win32gui.ClientToScreen(hwnd, (x, y))
    #x1, y1 = win32gui.ClientToScreen(hwnd, (x1 - x, y1 - y))
    #win32gui.ReleaseDC(hwnd, wDC)

    hwnd = win32gui.GetDesktopWindow()
    wDC = win32gui.GetWindowDC(hwnd)
    dcObj=win32ui.CreateDCFromHandle(wDC)
    cDC=dcObj.CreateCompatibleDC()
    dataBitMap = win32ui.CreateBitmap()
    dataBitMap.CreateCompatibleBitmap(dcObj, w, h)
    cDC.SelectObject(dataBitMap)
    cDC.BitBlt((0,0), (w, h), dcObj, (x, y), win32con.SRCCOPY)
    img_array = dataBitMap.GetBitmapBits(True)
    img = np.frombuffer(img_array, dtype='uint8')
    img.shape = (h, w, 4)
    dcObj.DeleteDC()
    cDC.DeleteDC()
    win32gui.ReleaseDC(hwnd, wDC)
    win32gui.DeleteObject(dataBitMap.GetHandle())

    return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)


def test1():

    keyboard.wait("s")
    width = 800
    height = 450
    #Image.fromarray(screenshot2()).save("C:/Users/HP/Desktop/scr.png")

    print("win32...")
    start = time()

    i = 100
    while i > 0:
        i-=1
        npimg = screenshot2(w=width, h=height)

    end = time() - start
    print(end)
    #Image.fromarray(npimg).save("C:/Users/HP/Desktop/scr2.png")



    print("win32...")
    start = time()

    i = 100
    while i > 0:
        i-=1
        npimg = screenshot(w=width, h=height)

    end = time() - start
    print(end)
    #Image.fromarray(npimg).save("C:/Users/HP/Desktop/scr1.png")


def test2():

    #cap = cv2.VideoCapture(0)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'MP4V') #(*'MP42')
    out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (800, 450))

    i = 1000

    while i>0:
        i-=1
        frame = screenshot2()

        out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release everything if job is finished
    out.release()
    cv2.destroyAllWindows()

"""print("PIL...")
start = time()

i = 100
while i > 0:
    i-=1
    npimg = pil_screenshot()

end = time() - start
print(end)
Image.fromarray(npimg).save("C:/Users/HP/Desktop/scr3.png")"""
