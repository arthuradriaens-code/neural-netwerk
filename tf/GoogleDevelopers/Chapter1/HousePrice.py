from tensorflow import keras
import numpy as  np

# find 50X + 50

model = keras.Sequential([keras.layers.Dense(units=1,input_shape=[1])])
model.compile(optimizer='sgd',loss='mean_squared_error')

xs = np.array([0,1,2,3,4,5], dtype=int)
ys = np.array([50,100,150,200,250,300], dtype=int)

model.fit(xs,ys,epochs=500)

print(model.predict([7]))
