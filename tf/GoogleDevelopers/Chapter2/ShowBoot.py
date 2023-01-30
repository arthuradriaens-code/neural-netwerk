import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt #plot the image

fashion_mnist = keras.datasets.fashion_mnist


(train_images, train_labels) , (test_images, test_labels) = fashion_mnist.load_data()

train_images = train_images / 255.0 #normalize
test_images = test_images / 255.0 

plt.imshow(test_images[0])
#print(test_labels[0])
#model.evaluate(test_images[0])
plt.show()
