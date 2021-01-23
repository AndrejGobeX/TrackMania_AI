# Dataset wrapper

import os
import pathlib
from PIL import Image
import Mods
import numpy as np

# rldu

def parse_output(string):
    r = int(string[0]) * 1.0
    l = int(string[1]) * 1.0
    d = int(string[2]) * 1.0
    u = int(string[3]) * 1.0
    return np.array([r, l, d, u])

def get_dataset(w, h, camera='third_person', mod=Mods.mod_neptune_crop, lines=None):
    path = './images/' + camera + '/'
    x_set = []
    y_set = []
    x_speed = []

    for folder in os.listdir(path):
        leaf_folder = path + folder + '/'
        parsed_y = parse_output(folder)
        for img in os.listdir(leaf_folder):
            img_path = leaf_folder + img
            
            # gets the speed
            speed = int(img.split('_')[-1][:-4])
            # opens image
            temp_img = Image.open(img_path)
            # apply mod
            if lines is not None:
                temp_img = mod(temp_img, w, h, no_lines=lines)
            else:
                temp_img = mod(temp_img, w, h) / 255.0
            # add to x_set
            x_set.append(temp_img)
            # add coding to y_set
            y_set.append(parsed_y)
            # add speed to x_speed
            x_speed.append(speed)

    x_set = np.array(x_set)
    y_set = np.array(y_set)
    x_speed = np.array(x_speed)
    
    p = np.random.permutation(len(x_set))
    
    x_set = x_set[p]
    y_set = y_set[p]
    x_speed = x_speed[p]

    x_set = (x_set, x_speed)

    dataset = (x_set, y_set)
    return dataset

# get_dataset(1400, 750)