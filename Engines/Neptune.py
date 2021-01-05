import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from PIL import Image
import Mods

class Neptune():

    def __init__(self, w=50, h=50, no_outputs=4, batch_size=None):
        
        self.name = 'neptune'
        self.checkpoint_path = checkpoint_path='./Checkpoints/' + self.name + '/ch.chpt'
        self.image_width = w
        self.image_height = h
        self.no_outputs = no_outputs
        self.batch_size = batch_size
        self.mod_function = Mods.mod_neptune_crop

        """self.model = keras.Sequential([
            #keras.layers.Input(shape=(self.image_height, self.image_width, 1)),
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
        ])"""

        self.model = self.assemble_model()

        self.model.compile(optimizer='adam',
            loss=keras.losses.MeanSquaredError(),
            metrics=['accuracy'])

        self.checkpoint_callback=keras.callbacks.ModelCheckpoint(
            filepath=self.checkpoint_path,
            save_weights_only=True,
            verbose=2
        )

    def assemble_model(self):

        image_input = keras.Input(
            shape=(self.image_height, self.image_width, 1),
            name='image_input'
        )

        speed_input = keras.Input(
            shape=(1),
            name='speed_input'
        )

        original = keras.layers.Conv2D(32,kernel_size=5,activation='relu')(image_input)
        original = keras.layers.MaxPool2D()(original)
        original = keras.layers.Dropout(0.4)(original)
        original = keras.layers.Conv2D(64,kernel_size=5,activation='relu')(original)
        original = keras.layers.MaxPool2D()(original)
        original = keras.layers.Dropout(0.4)(original)
        original = keras.layers.Flatten()(original)
        original = keras.layers.Dense(128, activation='relu')(original)
        original = keras.layers.Dropout(0.4)(original)

        speed = keras.layers.Dense(64, activation='relu')(speed_input)

        combined = keras.layers.concatenate([speed, original])
        combined = keras.layers.Dense(self.no_outputs, activation='sigmoid', name='keys')(combined)

        model = keras.Model(
            inputs=[image_input, speed_input],
            outputs=[combined]
        )

        return model

    
    def fit(self, train_input, train_output, no_epochs): #, validation_data=None):
        """self.model.fit(train_input, train_output, epochs=no_epochs,
            validation_data=validation_data,
            callbacks=[self.checkpoint_callback],
            batch_size=self.batch_size)"""
        
        self.model.fit(
            {'image_input': train_input[0], 'speed_input': train_input[1]},
            {'keys': train_output},
            epochs=no_epochs,
            batch_size=self.batch_size,
            callbacks=[self.checkpoint_callback]
        )

    def evaluate(self, test_input, test_output):
        self.model.evaluate(test_input, test_output, verbose=1)

    def predict(self, test_input):
        return self.model.predict(
            [test_input[0], test_input[1]]
        )

    def summary(self):
        self.model.summary()

    def load(self):
        try:
            self.model.load_weights(self.checkpoint_path)
        except:
            print('No checkpoint found')


# keras.utils.plot_model(Neptune().model, "multi_neptune.jpg", show_shapes=True)