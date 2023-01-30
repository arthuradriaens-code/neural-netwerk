# Idea
convolution is like filters in photoshop,  look at the diagram below:
![[figures/image1662.svg]]

The Top right 9 boxes is the images pixel representation and the red square of 9 numbers below it is the filter definition. Each value in this filter can be called a weight.
To calculate a new value for the "Current pixel" (i.e 192) we just multiply the pixel by it's weight, all the neighbours by their weight and add them all up:
Current pixel value: $192$
New Pixel value: $$\begin{align}&(-1*0) + (0*64) + (-2*128) + (0.2*48) + (4.5*192) +\\ &(-1.5*144) + (1.5*142) + (226*2) + (168*-3)\end{align}$$
Resulting is a transformed image.

E.g can be used to **emphasize** vertical/horizontal lines.


## code:
```python
for x in range(1,size_x-1):
    for y in range(1,size_y-1):
        convolution = 0.0
        convolution += (i[x-1,y-1] * filter[0][0])
        convolution += (i[x,y-1] * filter[0][1])
        convolution += (i[x+1,y-1] * filter[0][2])
        convolution += (i[x-1,y] * filter[1][0])
        convolution += (i[x,y] * filter[1][1])
        convolution += (i[x+1,y] * filter[1][2])
        convolution += (i[x-1,y+1] * filter[2][0])
        convolution += (i[x,y+1] * filter[2][1])
        convolution += (i[x+1,y+1] * filter[2][2])
        if (convolution<0):
            convolution = 0
        if (convolution>255):
            convolution=255
        i_transformed[x,y]=convolution
```
