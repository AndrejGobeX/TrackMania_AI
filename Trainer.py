# Training script
# import engine at the top of the script

from Engines import Neptune
import Mods, Dataset
from PIL import Image
import numpy as np

# image dimensions
image_width = 100
image_height = 100

no_epochs = 5000

# misc
train = True
load = True

# model = (your favorite engine goes here)
model = Neptune.Neptune(image_width, image_height, batch_size=10)

x, y = Dataset.get_dataset(image_width, image_height)

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