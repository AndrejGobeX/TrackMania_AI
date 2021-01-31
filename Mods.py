# Image modification functions

from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import numpy as np
import time
import os

def initial_crop(img, l, u, r, d):
    img = img.crop((l, u, img.size[0]-r, img.size[1]-d))
    return img

"""
def mod_neptune_crop(img, w, h):
    img = initial_crop(img, 100, 350, 100, 100).convert('L')
    img = img.resize((w, h), Image.ANTIALIAS)
    img = np.array(img).reshape((h, w, 1))
    return img

def mod_maniack_crop(img, w, h):
    img = initial_crop(img, 100, 350, 100, 100).convert('L')
    img = img.resize((w, h), Image.ANTIALIAS)
    img = np.array(img).reshape((h, w, 1))
    img = (img < 80) * 255.0
    return img
"""

def mod_neos(img, model):
    w = model.image_width
    h = model.image_height
    d = model.image_depth
    img = initial_crop(img, 0, img.size[1]//2, 0, img.size[1]//3)
    img = img.resize((w, h), Image.ANTIALIAS)
    if d==1:
        img = img.convert('L')
    #img = img.filter(ImageFilter.FIND_EDGES)
    img = np.array(img)
    return img.reshape((h, w, d))

def mod_road_mask_crop(img, model):
    w = model.image_width
    h = model.image_height
    img = initial_crop(img, 0, 470, 0, 300)
    img = img.resize((w, h), Image.ANTIALIAS).convert('L')
    img = ImageEnhance.Contrast(img).enhance(10)
    img = np.array(img)
    return img.reshape((h, w, 1))

def mod_edge(img, model):
    w = model.image_width
    h = model.image_height
    img = initial_crop(img, 0, img.size[1]//2, 0, img.size[1]//3)
    img = ImageEnhance.Contrast(img).enhance(2).convert('L')#.filter(ImageFilter.EDGE_ENHANCE_MORE)
    img = img.resize((w, h), Image.ANTIALIAS)
    img = np.array(img)
    img = (img < 50) * np.uint8(255)
    return img.reshape((h, w, 1))

def mod_shrink_n_measure(img, model, mod_fun=mod_road_mask_crop):
    no_lines = model.no_lines
    img_np = mod_fun(img, model)
    return find_walls(img_np, no_lines=no_lines)

def find_walls(img_np, no_lines=10, treshold=50):
    h, w, d = img_np.shape
    dx = w//no_lines

    end_points = []

    start_points = range(dx//2, w, dx)
    for start_point in start_points:
        distance = h - 1
        while distance >= 0:
            if img_np[distance][start_point] <= treshold: # pixel treshold
                break
            distance -= 1
        distance = h - distance - 1
        end_points.append(distance * 1.0 / h)
    
    return end_points

def run_inference(img_np, end_points):
    no_lines = len(end_points)
    h, w, d = img_np.shape
    dx = w//no_lines

    if d == 1:
        img_np = np.stack((img_np,)*3, axis=-1).reshape(h, w, 3)

    start_points = range(dx//2, w, dx)
    for start_point, end_point in zip(start_points, end_points):
        distance = end_point * h
        while distance > 0:
            i = int(h - distance)
            if i >= h:
                i = h - 1
            img_np[i][start_point][0] = 0
            img_np[i][start_point][1] = 255
            img_np[i][start_point][2] = 255
            distance -= 1
    
    return img_np