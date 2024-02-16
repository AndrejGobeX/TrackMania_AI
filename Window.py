import numpy as np
import win32gui
import win32ui
import win32con
import ctypes


class WindowInterface:
    def __init__(self, window_name="Trackmania"):
        self.window_name = window_name

        user32 = ctypes.windll.user32
        user32.SetProcessDPIAware()
        #ctypes.windll.shcore.SetProcessDpiAwareness(1)

        hwnd = win32gui.FindWindow(None, self.window_name)
        assert hwnd != 0, f"Could not find a window named {self.window_name}."

        while True:  # in case the window is reduced
            wr = win32gui.GetWindowRect(hwnd)
            cr = win32gui.GetClientRect(hwnd)
            if cr[2] > 0 and cr[3] > 0:
                break

        self.w_diff = wr[2] - wr[0] - cr[2] + cr[0]  # (16 on W10)
        self.h_diff = wr[3] - wr[1] - cr[3] + cr[1]  # (39 on W10)

        self.borders = (self.w_diff // 2, self.h_diff - self.w_diff // 2)

        self.x_origin_offset = - self.w_diff // 2
        self.y_origin_offset = 0

    def screenshot(self):
        hwnd = win32gui.FindWindow(None, self.window_name)
        assert hwnd != 0, f"Could not find a window named {self.window_name}."

        while True:  # avoids crashes when the window is reduced
            x, y, x1, y1 = win32gui.GetWindowRect(hwnd)
            w = x1 - x - self.w_diff
            h = y1 - y - self.h_diff
            if w > 0 and h > 0:
                break

        hdc = win32gui.GetWindowDC(hwnd)
        dc = win32ui.CreateDCFromHandle(hdc)
        memdc = dc.CreateCompatibleDC()
        bitmap = win32ui.CreateBitmap()
        bitmap.CreateCompatibleBitmap(dc, w, h)
        oldbmp = memdc.SelectObject(bitmap)
        memdc.BitBlt((0, 0), (w, h), dc, self.borders, win32con.SRCCOPY)
        bits = bitmap.GetBitmapBits(True)
        img = (np.frombuffer(bits, dtype='uint8'))
        img.shape = (h, w, 4)
        memdc.SelectObject(oldbmp)  # avoids memory leak
        win32gui.DeleteObject(bitmap.GetHandle())
        memdc.DeleteDC()
        win32gui.ReleaseDC(hwnd, hdc)
        return img

    def move_and_resize(self, x=1, y=0, w=800, h=450):
        x += self.x_origin_offset
        y += self.y_origin_offset
        w += self.w_diff
        h += self.h_diff
        hwnd = win32gui.FindWindow(None, self.window_name)
        assert hwnd != 0, f"Could not find a window named {self.window_name}."
        win32gui.MoveWindow(hwnd, x, y, w, h, True)


# def profile_screenshot():
#     from pyinstrument import Profiler
#     pro = Profiler()
#     window_interface = WindowInterface("Trackmania")
#     pro.start()
#     for _ in range(5000):
#         snap = window_interface.screenshot()
#     pro.stop()
#     pro.print(show_all=True)

if __name__ == "__main__":
    import cv2
    import time
    wi = WindowInterface()
    wi.move_and_resize(w=800, h=450)
    times = []
    while True:
        start = time.time()
        sch = wi.screenshot()
        #sch = cv2.cvtColor(sch, cv2.COLOR_RGBA2GRAY)
        #sch = cv2.resize(sch, (80, 45))
        #sch = cv2.convertScaleAbs(sch, alpha=1.7, beta=0)
        times.append(time.time() - start)
        cv2.imshow("win", sch)
        if cv2.waitKey(1) == ord('q'):
            break
    print(sch.shape)
    print(sum(times)/len(times))
    cv2.destroyAllWindows()
