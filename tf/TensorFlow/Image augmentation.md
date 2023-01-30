# Image augmentation

## Problem
overfitting

## Solution: Image augmentation

say cat gets recognized by 2 ears pointing upwards, rotate the image.

### code:

```python
train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40, #40° left or right
        width_shift_range=0.2, #20%: move subject
        height_shift_range=0.2,
        shear_range=0.2, #e.g lying down human from standing
        zoom_range=0.2, #e.g no legs
        horizontal_flip=True, #left hand to right hand raised
        fill_mode='nearest' #fill in lost parts of the image
        )
```

