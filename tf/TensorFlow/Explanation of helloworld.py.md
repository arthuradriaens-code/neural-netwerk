## First line of code
model = keras.Sequential([keras.layers.Dense(units=1,input_shape=[1])])

Keras is a framework in tensorflow, Sequential tells us that we're going to define a sequence of layers e.g: 
![[figures/5a7af09d-digits0-1148441270.png]]
our figure has 4 layers, in our code we have only have 1 layer, dense means that every neuron is connected to every other neuron. We also only have 1 neuron in
our  code as units=1. Ours thus looks like this:

![[figures/path111.svg]]

## Second line of code

model.compile(optimizer='sgd',loss='mean_squared_error')

this is the training algorithm we choose. sgd: Stochastic gradient descend.

## Learning

model.fit(xs,ys,epochs=500)

(obvious what it does)