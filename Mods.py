# Image modification functions

from PIL import Image
import numpy as np


def mod_neptune_crop(img, w, h):
    # img -- PIL image
    img = img.crop((100, 350, img.size[0]-100, img.size[1]-100)).convert('L')
    img = img.resize((w, h), Image.ANTIALIAS)
    img = np.array(img).reshape((h, w, 1))
    return img


"""img = Image.open('./images/third_person/0101/5_172.jpg')
img = img.crop((100, 350, img.size[0]-100, img.size[1]-100)).convert('L')
img = img.resize((100, 100), Image.ANTIALIAS)
img.show()"""