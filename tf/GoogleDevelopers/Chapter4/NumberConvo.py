import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt #plot the image

mnist = keras.datasets.mnist

class myCallback(keras.callbacks.Callback): #code given to fit to stop when 99%
    def on_epoch_end(self,epoch,logs={}):
        if (logs.get('accuracy')>0.998):
            print("\nReached 99.8% accuracy so cancelling training!")
            self.model.stop_training = True

callbacks=myCallback()

(train_images, train_labels) , (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape(60000,28,28,1)
train_images = train_images / 255.0 #normalize
test_images = test_images.reshape(10000,28,28,1)
test_images = test_images / 255.0 

model = keras.Sequential([
    keras.layers.Conv2D(32,(3,3), activation='relu', input_shape=(28,28,1)),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(train_images,train_labels,epochs=20,callbacks=[callbacks])

model.evaluate(test_images,test_labels)

#plt.imshow(test_images[0])
#plt.show()

