import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt #plot the image

fashion_mnist = keras.datasets.fashion_mnist


(train_images, train_labels) , (test_images, test_labels) = fashion_mnist.load_data()

train_images = train_images / 255.0 #normalize
test_images = test_images / 255.0 

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(train_images,train_labels,epochs=5)

model.evaluate(test_images,test_labels)

#plt.imshow(test_images[0])
#print(test_labels[0])
#model.evaluate(test_images[0])
#plt.show()

classifications = model.predict(test_images)
print(classifications[0])
