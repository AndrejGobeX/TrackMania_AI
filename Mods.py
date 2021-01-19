# Image modification functions

from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
import time
import os

def initial_crop(img, l, u, r, d):
    img = img.crop((l, u, img.size[0]-r, img.size[1]-d))
    return img

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

def mod_road_mask_crop(img, w, h):
    img = initial_crop(img, 0, 0, 0, 100).convert('L')
    img = img.resize((w, h), Image.ANTIALIAS)
    img = ImageEnhance.Contrast(img).enhance(10)
    # img = img.filter(ImageFilter.EDGE_ENHANCE)
    img = np.array(img).reshape((h, w, 1))
    return img

letter = 't'
i = 100
for name in os.listdir('./images/third_person/1001/'):
    if i==0:
        break
    i-=1
    if name[0] == '6' or name[0] == '7':
        img = './images/third_person/1001/' + name
        img = Image.open(img)
        start = time.time()
        img = mod_road_mask_crop(img, 50, 50).reshape((50, 50))
        end = time.time()
        print(end - start)
        img = Image.fromarray(img).convert('L')
        img.save('./images/third_person/1001/'+letter+name)