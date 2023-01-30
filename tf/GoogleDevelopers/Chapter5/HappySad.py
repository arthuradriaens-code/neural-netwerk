import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt #plot the image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop

class myCallback(keras.callbacks.Callback): #code given to fit to stop when 99%
    def on_epoch_end(self,epoch,logs={}):
        if (logs.get('accuracy')>0.999):
            print("\nReached 99.9% accuracy so cancelling training!")
            self.model.stop_training = True

callbacks=myCallback()

train_datagen = ImageDataGenerator(rescale=1.255)
train_generator = train_datagen.flow_from_directory(
    'happysadimages/',
	target_size=(300,300),
	batch_size=8,
	class_mode='binary'
)


model = keras.Sequential([
    keras.layers.Conv2D(16,(3,3), activation='relu', input_shape=(300,300,3)),
    #3 bytes per pixel: RGB
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Conv2D(32,(3,3), activation='relu'),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Conv2D(64,(3,3), activation='relu'),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.summary()

model.compile(optimizer=RMSprop(learning_rate=0.001),loss='binary_crossentropy',metrics=['accuracy'])

model.fit(train_generator,epochs=15,verbose=2,callbacks=[callbacks],steps_per_epoch=10)
