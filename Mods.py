# Image modification functions

from PIL import Image, ImageFilter
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
    img = initial_crop(img, 300, 350, 300, 100).convert('L')
    img = img.resize((w, h), Image.ANTIALIAS)
    img = img.filter(ImageFilter.EDGE_ENHANCE_MORE)
    img = np.array(img).reshape((h, w, 1))
    return img

letter = 't'
for name in os.listdir('./images/third_person/1001/'):
    if name[0] == '6' or name[0] == '7':
        img = './images/third_person/1001/' + name
        img = Image.open(img)
        start = time.time()
        img = mod_road_mask_crop(img, 30, 30).reshape((30, 30))
        end = time.time()
        print(end - start)
        img = Image.fromarray(img).convert('L')
        img.save('./images/third_person/1001/'+letter+name)
"""img = Image.open('./images/third_person/0101/6_306_191.jpg')

start = time.time()
img = img.crop((100, 350, img.size[0]-100, img.size[1]-100)).convert('L')

img = np.array(img)

img = (img < 70)

img = Image.fromarray(img)
img = img.resize((50, 50), Image.ANTIALIAS)
img.show()
img = np.array(img).reshape(50, 50, 1)
end = time.time()
print(end-start)
img = Image.open('./images/third_person/0101/6_306_191.jpg')
start = time.time()
img = img.crop((100, 350, img.size[0]-100, img.size[1]-100)).convert('L')
img = img.resize((50, 50), Image.ANTIALIAS)

img = np.array(img)

img = (img < 80) * 255.0

img = Image.fromarray(img)
img.show()
img = np.array(img).reshape(50, 50, 1)
end = time.time()
print(end-start)
img = Image.open('./images/third_person/0101/6_306_191.jpg')
img = mod_maniack_crop(img, 50, 50)
img = Image.fromarray(img.reshape(50, 50))
img.show()"""