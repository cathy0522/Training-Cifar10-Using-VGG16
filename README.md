---
title: Cifar10 classifier using VGG16
tags: Python, TensorFlow, keras
---

## Description
This is a practice project that using Python with TensorFlow.

## Requirements
* Python==3.7.0
* opencv-contrib-python==3.4.2.17
* matplotlib==3.1.1
* numpy==1.18.5

---

## Show the results
### 1. Show train image
Show the Load Cifar10 training dataset and randomly show 10 images and labels.
![](https://i.imgur.com/9m0CwrT.png)

### 2. Show training hyperparameters 
![](https://i.imgur.com/3lobTbM.png)

### 3. Show VGG16 model
![](https://i.imgur.com/xu8k7Ys.png)

Or you can show this: 
```model=
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 32, 32, 64)        1792      
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 32, 32, 64)        36928     
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 16, 16, 64)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 16, 16, 128)       73856     
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 16, 16, 128)       147584    
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 8, 8, 128)         0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 8, 8, 256)         295168    
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 8, 8, 256)         590080    
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 8, 8, 256)         590080    
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 4, 4, 256)         0         
_________________________________________________________________
conv2d_7 (Conv2D)            (None, 4, 4, 512)         1180160   
_________________________________________________________________
conv2d_8 (Conv2D)            (None, 4, 4, 512)         2359808   
_________________________________________________________________
conv2d_9 (Conv2D)            (None, 4, 4, 512)         2359808   
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 2, 2, 512)         0         
_________________________________________________________________
conv2d_10 (Conv2D)           (None, 2, 2, 512)         2359808   
_________________________________________________________________
conv2d_11 (Conv2D)           (None, 2, 2, 512)         2359808   
_________________________________________________________________
conv2d_12 (Conv2D)           (None, 2, 2, 512)         2359808   
_________________________________________________________________
max_pooling2d_4 (MaxPooling2 (None, 1, 1, 512)         0         
_________________________________________________________________
flatten (Flatten)            (None, 512)               0         
_________________________________________________________________
dense (Dense)                (None, 4096)              2101248   
_________________________________________________________________
dense_1 (Dense)              (None, 4096)              16781312  
_________________________________________________________________
dense_2 (Dense)              (None, 2)                 8194      
=================================================================
Total params: 33,605,442
Trainable params: 33,605,442
Non-trainable params: 0
_________________________________________________________________

```

### 4. Accuracy and training loss
![](https://i.imgur.com/VJdKWLj.png)

![](https://i.imgur.com/QhVqsxY.png)

### 5. Show test images
Random choose test image=1000 : 

![](https://i.imgur.com/7g3KjVO.png)

![](https://i.imgur.com/IykFJMG.png)

