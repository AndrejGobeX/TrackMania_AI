import cv2
import numpy as np
import math


def armin(tab):
    nz = np.nonzero(tab)[0]
    if len(nz) != 0:
        return nz[0].item()
    else:
        return len(tab) - 1
    

def rotate_image(image, image_center, angle):
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[:-1])
    return result


class TrackVisualizer():


    VIEW_WIDTH = 200
    VIEW_HEIGHT = 200
    VIEW_CENTER = (VIEW_WIDTH//2, VIEW_HEIGHT//2)
    MAX_DISTANCE = math.hypot(VIEW_HEIGHT,VIEW_WIDTH)/2
    CONTACT_THRESHOLD = 0.035
    

    def __init__(self, map):
        self._set_axis_lidar()
        self.black_threshold = [55,55,55]
        self.min_x = int(np.min(map[:,[0,2]])) - TrackVisualizer.VIEW_WIDTH//2
        self.max_x = int(np.max(map[:,[0,2]])) + TrackVisualizer.VIEW_WIDTH//2
        self.min_y = int(np.min(map[:,[1,3]])) - TrackVisualizer.VIEW_HEIGHT//2
        self.max_y = int(np.max(map[:,[1,3]])) + TrackVisualizer.VIEW_HEIGHT//2
        
        self.track = np.ones((int(self.max_y - self.min_y), int(self.max_x - self.min_x), 3), dtype=np.uint8)*255
        
        for cp0, cp1 in zip(map[:-1], map[1:]):
            rx0 = int(cp0[0]) - self.min_x
            rx1 = int(cp1[0]) - self.min_x
            lx0 = int(cp0[2]) - self.min_x
            lx1 = int(cp1[2]) - self.min_x

            ry0 = int(cp0[1]) - self.min_y
            ry1 = int(cp1[1]) - self.min_y
            ly0 = int(cp0[3]) - self.min_y
            ly1 = int(cp1[3]) - self.min_y

            cv2.line(self.track, [rx0, ry0], [rx1, ry1], color=[0,0,0], thickness=2)
            cv2.line(self.track, [lx0, ly0], [lx1, ly1], color=[0,0,0], thickness=2)
        
    
    def _set_axis_lidar(self):
        min_dist = 0
        list_ax_x = []
        list_ax_y = []
        for angle in range(90, 280, 15):
            axis_x = []
            axis_y = []
            x = TrackVisualizer.VIEW_CENTER[1]
            y = TrackVisualizer.VIEW_CENTER[0]
            dx = math.cos(math.radians(angle))
            dy = math.sin(math.radians(angle))
            lenght = False
            dist = min_dist
            while not lenght:
                newx = int(x + dist * dx)
                newy = int(y + dist * dy)
                if newx <= 0 or newy <= 0 or newy >= TrackVisualizer.VIEW_WIDTH - 1 or newx >= TrackVisualizer.VIEW_HEIGHT - 1:
                    lenght = True
                    list_ax_x.append(np.array(axis_x))
                    list_ax_y.append(np.array(axis_y))
                else:
                    axis_x.append(newx)
                    axis_y.append(newy)
                dist = dist + 1
        self.list_axis_x = list_ax_x
        self.list_axis_y = list_ax_y


    def lidar_13(self, img, show=False):
        distances = []
        if show:
            color = (255, 255, 0)
            thickness = 1
        for axis_x, axis_y in zip(self.list_axis_x, self.list_axis_y):
            index = armin(np.all(img[axis_x, axis_y] < self.black_threshold, axis=1))
            if show:
                img = cv2.line(img, TrackVisualizer.VIEW_CENTER, (axis_y[index], axis_x[index]), color, thickness)
            index = np.float32(index)
            distances.append(index)
        res = np.array(distances, dtype=np.float32)
        if show:
            cv2.imshow("Environment", img)
            cv2.waitKey(1)
        return res
    

    def lidar(self, location, θ, show=False):
        x = int(location[0]) - self.min_x
        y = int(location[1]) - self.min_y
        img = self.track[
            y - TrackVisualizer.VIEW_HEIGHT//2 : y + TrackVisualizer.VIEW_HEIGHT//2,
            x - TrackVisualizer.VIEW_WIDTH//2 : x + TrackVisualizer.VIEW_WIDTH//2,
            :
        ].copy()
        
        if show:
            cv2.circle(img, TrackVisualizer.VIEW_CENTER, 1, color=[255,150,0], thickness=2)
        img = rotate_image(img, TrackVisualizer.VIEW_CENTER, math.degrees(θ + math.pi/2))
        return self.lidar_13(img, show=show) / TrackVisualizer.MAX_DISTANCE

