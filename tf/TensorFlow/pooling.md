# Goal
Extract features while removing extraneous information.

# Concept

Keep biggest value in block of 2X2 yielding only 1 block.

E.g 4X4 $\rightarrow$ 2X2

## code
```python

new_x = int(size_x/2)
new_y = int(size_y/2)
newImage = np.zeros((new_x,new_y))

for x in range(0,size_x,2):
    for y in range(0,size_y,2):
    
        pixels = []
        pixels.append(i_transformed[x,y])
        pixels.append(i_transformed[x+1,y])
        pixels.append(i_transformed[x,y+1])
        pixels.append(i_transformed[x+1,y+1])
        pixels.sort(reverse=True)
        newImage[int(x/2),int(y/2)] = pixels[0]
```


