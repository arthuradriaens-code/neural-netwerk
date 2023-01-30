import gzip
import numpy as np
import matplotlib.pyplot as plt

f = gzip.open('/home/arthur/Documents/code/ML/data/train-images-idx3-ubyte.gz','r')
image_size = 28
num_images = 50000
f.read(16)
buf = f.read(image_size * image_size * num_images)
data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
data = data.reshape(num_images, image_size, image_size, 1)
image = np.asarray(data[145]).squeeze()
plt.imshow(image)
plt.show()

r = gzip.open('/home/arthur/Documents/code/ML/data/train-labels-idx1-ubyte.gz','r')
r.read(8)
buf = r.read(num_images)
label = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
print(label[145])
print(type(image))
