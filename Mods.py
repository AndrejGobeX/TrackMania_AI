# Image modification functions

from PIL import Image, ImageFilter, ImageEnhance, ImageOps
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
    img = initial_crop(img, 0, 460, 0, 400)
    img = img.resize((w, h), Image.ANTIALIAS).convert('L')
    img = ImageEnhance.Contrast(img).enhance(10)
    # img = img.filter(ImageFilter.FIND_EDGES)
    # img = mask_wrapper(img, black, mask) # TODO: nesto
    img = np.array(img)
    # img[0:h//2, 0:w] = np.zeros((h//2, w))
    return img.reshape((h, w, 1))

def mask_wrapper(img, black, mask):
    return Image.composite(img, black, mask)

"""black = Image.open('./images/mask3.png').convert('L').resize((160, 90), Image.ANTIALIAS)
mask = initial_crop(ImageOps.invert(Image.open('./images/mask.png').convert('L')), 0, 200, 0, 0).resize((160, 90), Image.ANTIALIAS)

letter = 't'
i = 100
for name in os.listdir('./images/first_person/0000/'):
    if i==0:
        break
    i-=1
    if name[0] == '1' or name[0] == '1':
        img = './images/first_person/0000/' + name
        img = Image.open(img)
        start = time.time()
        img = mod_road_mask_crop(img, 50, 50).reshape((50, 50))
        end = time.time()
        print(end - start)
        img = Image.fromarray(img).convert('L')
        img.save('./images/first_person/0000/'+letter+name)"""