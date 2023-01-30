Structure like this:

Images
|
--Training----horses: 1.jpg,2.jpg,3.jpg
|                   |
|                   --humans: Polina.jpg, Freddie.jpg, Arthur.jpg
|
--Validation--horses: SomeHorse.jpg
                     |
                     --humans: SomeNingen.jpg

in directory can easily be labeled with the image data generator:

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator
```

This can then be normalized as follows:

```python
train_datagen = ImageDataGenerator(rescale=1.255)
```

Now "flowing" them out of the directory:

```python
train_generator = train_datagen.flow_from_directory(
	train_dir,
	target_size=(300,300)
	batch_size=128,
	class_mode='binary'
)
```

train_dir is directory path of training data, target size is as not all might be same size. Batch size for training, class_mode is binary as 2 classes otherwise: categorical.
Same for validation:

```python
test_datagen = ImageDataGenerator(rescale=1.255)

test_generator = test_datagen.flow_from_directory(
	test_dir,
	target_size=(300,300)
	batch_size=32,
	class_mode='binary'
)
```

```python
model.fit(train_generator,epochs=15,validation_data=validation_generator,verbose=2,callbacks=[callbacks],steps_per_epoch=8,validation_steps=8)
```

Rule of thumb for steps per epoch: amount of items in dataset / batch size.

## Seperate files into training and test:

```shell
ls Dog/ | gshuf -n 11250 | xargs -I {} mv Dog/{} training/Dog/
```

