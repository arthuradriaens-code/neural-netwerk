## model

model = keras.Sequential([

    keras.layers.Flatten(input_shape=(28,28)),

    keras.layers.Dense(128, activation=tf.nn.relu),

    keras.layers.Dense(10, activation=tf.nn.softmax)

])

input: Flatten: take the rectangular (28x28) shape into 1D array.

First dense: 128 neurons; activation tf.nn.relu, relu: if x<0 -> x = 0.

output Dense: 10 neurons; tf.nn.softmax, softmax: commonly seen on final layer when multiple categories: helps you find the most likely candidate
(instead of 0.02,0.01,0.02,..,0.978 it will return 0,0,0,...,1 ).

## compile
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy')

When categories: something categorical and adam "good for this"..?

## learning

model.fit(train_images,train_labels,epochs=5)    

model.evaluate(test_images, test_labels)
