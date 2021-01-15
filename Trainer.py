# Training script
# python Trainer.py engine_name [no_epochs]
# python Trainer.py Neptune 20

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import Mods, Dataset
from PIL import Image
import numpy as np
import sys

# first parameter is the name of the engine class

if len(sys.argv) < 2:
    print('Missing engine class.')
    exit()

# get the engine class

model_name = sys.argv[1]
engine_module = __import__('Engines.' + model_name, fromlist=[model_name])
engine_class = getattr(engine_module, model_name)

model = engine_class(batch_size=8)

# image dimensions
image_width = model.image_width
image_height = model.image_height

# second parameter is the number of epochs, if missing defaults to 100
no_epochs = 2
if len(sys.argv) == 3:
    no_epochs = int(sys.argv[2])

# third parameter is the camera, if missing defaults to third_person
# only third_person and first_person are available
camera = 'third_person'
if len(sys.argv) == 4:
    camera = sys.argv[3]

# misc
train = True
load = True

x, y = Dataset.get_dataset(image_width, image_height, mod=model.mod_function, camera=camera)

size = len(x)
train_percentage = size*9//10
test_percentage = size - train_percentage

x_train = x #x[:train_percentage]
y_train = y #y[:train_percentage]

#x_test = x[train_percentage:len(x)]
#y_test = y[train_percentage:len(y)]

if load:
    model.load()
if train:
    model.fit(x_train, y_train, no_epochs)

model.summary()

"""for i in range(len(x)):
    print("image: "+str(i))

    if(int(y[i][0])==0 and int(y[i][1])==0.0):
        continue
    prediction = model.predict(x[i:i+1])
    
    print(prediction)
    print((prediction[0] > 0.1))
    print(y[i])
    print()
"""