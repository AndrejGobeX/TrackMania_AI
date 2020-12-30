# Training script
# import engine at the top of the script

from Engines import Neptune
import Mods, Dataset
from PIL import Image
import numpy as np

model = Neptune.Neptune(100, 100)

x, y = Dataset.get_dataset(100, 100)

size = len(x)
train_percentage = size*9//10
test_percentage = size - train_percentage

x_train = x[:train_percentage]
y_train = y[:train_percentage]

x_test = x[train_percentage:len(x)]
y_test = y[train_percentage:len(y)]

# model.fit(x_train, y_train, 10, (x_test, y_test))
model.load()

for i in range(len(x)):
    print("image: "+str(i))

    if(int(y[i][0])==0 and int(y[i][1])==0.0):
        continue
    prediction = model.predict(x[i:i+1])
    
    print(prediction)
    print((prediction[0] > 0.1))
    print(y[i])
    print()
