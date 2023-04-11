import math
import numpy as np
import cv2


def armin(tab):
    nz = np.nonzero(tab)[0]
    if len(nz) != 0:
        return nz[0].item()
    else:
        return len(tab) - 1


class Lidar:
    def __init__(self, im):
        self._set_axis_lidar(im)
        self.black_threshold = [55,55,55]


    def _set_axis_lidar(self, im):
        h, w, _ = im.shape
        self.h = h
        self.w = w
        self.road_point = (44*h//49, w//2)
        min_dist = 20
        list_ax_x = []
        list_ax_y = []
        for angle in range(90, 280, 10):
            axis_x = []
            axis_y = []
            x = self.road_point[0]
            y = self.road_point[1]
            dx = math.cos(math.radians(angle))
            dy = math.sin(math.radians(angle))
            lenght = False
            dist = min_dist
            while not lenght:
                newx = int(x + dist * dx)
                newy = int(y + dist * dy)
                if newx <= 0 or newy <= 0 or newy >= w - 1:
                    lenght = True
                    list_ax_x.append(np.array(axis_x))
                    list_ax_y.append(np.array(axis_y))
                else:
                    axis_x.append(newx)
                    axis_y.append(newy)
                dist = dist + 1
        self.list_axis_x = list_ax_x
        self.list_axis_y = list_ax_y


    def lidar_20(self, img, show=False):
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        h, w, _ = img.shape
        if h != self.h or w != self.w:
            self._set_axis_lidar(img)
        distances = []
        if show:
            color = (255, 0, 0)
            thickness = 4
        for axis_x, axis_y in zip(self.list_axis_x, self.list_axis_y):
            index = armin(np.all(img[axis_x, axis_y] < self.black_threshold, axis=1))
            if show:
                img = cv2.line(img, (self.road_point[1], self.road_point[0]), (axis_y[index], axis_x[index]), color, thickness)
            index = np.float32(index)
            distances.append(index)
        res = np.array(distances, dtype=np.float32)
        if show:
            cv2.imshow("Environment", img)
            cv2.waitKey(1)
        return res


if __name__ == "__main__":
    pass
