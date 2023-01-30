We'll now implement the things discussed in [[convolutional neural network]] and [[pooling]]:

```python
model = keras.Sequential([
    keras.layers.Conv2D(64,(3,3), activation='relu', input_shape=(28,28,1)),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Conv2D(64,(3,3), activation='relu'),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Flatten(), 
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)

])
```
3 last lines are the same, 64: 64 filters; each 3X3 random and further on optimised.
pooling then and then again.

To get an overview:

```model.summary()```

giving:

 Layer (type)                Output Shape              Param #   

=================================================================

 conv2d (Conv2D)             (None, 26, 26, 64)        640       

 max_pooling2d (MaxPooling2D  (None, 13, 13, 64)       0         

 )                                                               

 conv2d_1 (Conv2D)           (None, 11, 11, 64)        36928     

 max_pooling2d_1 (MaxPooling  (None, 5, 5, 64)         0         

 2D)                                                             

 flatten (Flatten)           (None, 1600)              0         

 dense (Dense)               (None, 128)               204928    

 dense_1 (Dense)             (None, 10)                1290      

=================================================================

Total params: 243,786

Trainable params: 243,786

Non-trainable params: 0

_________________________________________________________________

why 26X26 insead of 28X28? say e.g top left pixel or top pixels dont have all the neighbours needed (8) so these need to be removed.

Each filter 9 values + bias $\rightarrow$ $10*64$ = 640 parameters

