# Image modification functions

from PIL import Image
import numpy as np
import time


def mod_neptune_crop(img, w, h):
    # img -- PIL image
    img = img.crop((100, 350, img.size[0]-100, img.size[1]-100)).convert('L')
    img = img.resize((w, h), Image.ANTIALIAS)
    img = np.array(img).reshape((h, w, 1))
    return img


"""img = Image.open('./images/third_person/0101/6_54_112.jpg')
start = time.time()
img = img.crop((100, 350, img.size[0]-100, img.size[1]-100)).convert('L')

img = np.array(img)

img = (img > 180) * img

img = Image.fromarray(img)
img = img.resize((50, 50), Image.ANTIALIAS)
img.show()
end = time.time()
print(end-start)"""