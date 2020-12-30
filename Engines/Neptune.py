import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from PIL import Image

class Neptune():

    def __init__(self, w, h, no_outputs=4):
        
        self.name = 'neptune'
        self.checkpoint_path = checkpoint_path='./Checkpoints/' + self.name + '/ch.chpt'
        self.image_width = w
        self.image_height = h
        self.no_outputs = no_outputs

        self.model = keras.Sequential([
            keras.layers.Conv2D(32,kernel_size=5,activation='relu',
                input_shape=(self.image_height, self.image_width, 1)),
            keras.layers.MaxPool2D(),
            keras.layers.Dropout(0.4),
            keras.layers.Conv2D(64,kernel_size=5,activation='relu'),
            keras.layers.MaxPool2D(),
            keras.layers.Dropout(0.4),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.4),
            keras.layers.Dense(self.no_outputs, activation='sigmoid')
        ])

        self.model.compile(optimizer='adam',
            loss=keras.losses.MeanSquaredError(),
            metrics=['accuracy'])

        self.checkpoint_callback=keras.callbacks.ModelCheckpoint(
            filepath=self.checkpoint_path,
            save_weights_only=True,
            verbose=2
        )
    
    def fit(self, train_input, train_output, no_epochs, validation_data=None):
        self.model.fit(train_input, train_output, epochs=no_epochs,
            validation_data=validation_data,
            callbacks=[self.checkpoint_callback])

    def evaluate(self, test_input, test_output):
        self.model.evaluate(test_input, test_output, verbose=1)

    def predict(self, test_input):
        return self.model.predict(test_input)

    def summary(self):
        self.model.summary()

    def load(self):
        self.model.load_weights(self.checkpoint_path)
