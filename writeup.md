# Project: Follow-Me

---

[//]: # (Image References)

[image_cover]: ./docs/misc/image_cover.png
[image_01]: ./docs/misc/image_01.PNG
[image_02]: ./docs/misc/image_02.png
[image_03]: ./docs/misc/image_03.png



## Introduction

The objective of this project was to design a deep neural network allowing a quadcopter to locate a target of interest (person) in a series of images captured with it's camera. Using this information, the quadcopter is able to follow around the target in an environment populated with other virtual humans and objects. 

The image below shows the quadcopter following the target of interest, and the images processed by the neural network (right -"pyqtgraph"). The target of interest is denoted by a dark purple silhouette, while the other people are denoted by green silhouettes. 

![alt text][image_cover]

## Network architecture

For this project we are interested not only identifying the target, but also determining where in the picture the target is located so that the quadcopter can adjust its course accordingly and follow. Thus, the spatial information in the image matters. Deep neural networks with fully connected layers or convolutional neural networks do not preserve this information. However, fully convolutional neural networks (FCNs) do.

FCNs allow us to extract features with different levels of complexity from images, and the use these features to segment the images into meaningful pieces. Thus, an FCN was used to segment that quadcopter images into three categories: background, other people, hero (target of interest).

In general, FCNs are composed of 3 parts: Encoders, a 1x1 convolution, and decoders.


![alt text][image_01]


The encoders basically breakdown the input images into key features: Starting with simple features (such as basic shapes in the first layers) and moving on to more complex features (or combinations of simple shapes in the last layers). With each encoder, the resolution of the original image is reduced and the number of extracted features increases (as shown by the increase in depth of each layer).
Each of these layers is the result of convolving a filter over that layer's input. The parameters or weights of these filters are learned during training and they determine what kind of features are extracted.


Next, a 1x1 convolution is applied on the output of the last encoder. This convolution consists in applying 1x1 filters (kernels),  which results in a new set of layers with the same width and height as the layers of the last encoder. 
This convolution is preferred for image segmentation (instead of fully connected layers) because it preserves spatial information and also allows the network to work with images of different sizes. Fully connected layers flatten the output of the last encoder, thus they are better suited for classification tasks where spatial information is not important and images are the same size.


Finally, the objective of the decoders are to map the features extracted by the encoders into an image with the same resolution as the input image. This is done by fusing (concatenating) various lower and higher resolution layers. For instance, in the image above the input to *Decoder 1* is upsampled and then fused with the layers of *Encoder 2*. This process is repeated until arriving to an image with the same resolution as the input image but segmented into three categories (3 channels): background, other people, and hero.


For this project the selected architechture is composed of the following elements:


* An extra convolutional layer (depth 16) after the input image and before the encoders for later use in layer fusion. This layer extracts features, but preserves the original image size. 

* 3 encoders (depths 32, 64, 128)

* 1 x 1 convolution (depth 256)

* 3 decoders (depths 128, 64, 32)

* Output layer (depth 3) with the same resolution as the last decoder layer, but with 3 channels for: background, other people, and hero.

Initially, I began with an architecture consisting of a first convolutional layer, 2 encoders, a 1 x 1 convolutional layer, 2 decoders, and an output layer (all with small depths). However, after trying different combinations of hyperparameters the performance (or final score) was stuck at around 0.35. 

This revealed that this architecture was too simple and didn't capture enough features to correctly discriminate between the desired classes.
Thus, I decided to go for a more complex architecture adding one more encoder and decoder and increasing the depth of each layer, which increased the final score.




## Tuning Hyperparameters

The performance or the ability of the network to correctly segment the image into the three classes of interest, also depends on the network's hyperparameters: batch size, learning rate, number of epochs, steps per epoch, validation steps and workers.

I selected the values for these parameters through manual tuning, aiming for a good functionality (final score above the required threshold: 0.4) and simplicity:

``` python
learning_rate = 0.005 
batch_size = 32
num_epochs = 20
steps_per_epoch = 200
validation_steps = 50
workers = 2
``` 

* Batch size: As stated [here](https://machinelearningmastery.com/gentle-introduction-mini-batch-gradient-descent-configure-batch-size/), usually the batch size is tuned with respect to the computational architecture where the network will be executed (e.g. memory requirement of the GPU or CPU). I tried batch sizes from 32 to 128 with my first architecture, and the performance didn't change much, so I selected a batch size of 32.

* Learning rate: I tried different learning rates, from 0.1 to 0.005. I noticed that as the learning rate decreased, the performance of the network increased. I selected a learning rate of 0.005 since it yielded a score above the required score (0.4).

* Number of epochs: I began with 10 epochs, which resulted in a score already above the required score. However, I wanted to verify if the performance would improve with more epochs. I doubled the epochs to 20 and it did. Therefore, I selected 20 as the number of epochs.

* Step per epoch, validation steps and workers: These parameters were left with their default values since, together with the above tuned parameters, they yielded a final score above the required one. 



## Implementation

The previously described network can be found in the *model_training.ipynb* notebook or *model_training.html* (in the *code/* directory). The final weights of the model can be found in the *model_weights5* file (in the *data/weights/* directory).

The encoder blocks were implemented as separable convolutions:


``` python
def encoder_block(input_layer, filters, strides):
    
    # TODO Create a separable convolution layer using the separable_conv2d_batchnorm() function.
    output_layer = separable_conv2d_batchnorm(input_layer, filters, strides)
    
    return output_layer
    
``` 



The decoder blocks consisted in upsampling the lowest resolution layer by a factor of 2, concatenating the lower and higher resolution layers, and applying a separable convolution to the result of this concatenation:

``` python
def decoder_block(small_ip_layer, large_ip_layer, filters):
    
    # TODO Upsample the small input layer using the bilinear_upsample() function.
    small_layer_up = bilinear_upsample(small_ip_layer)
    # TODO Concatenate the upsampled and large input layers using layers.concatenate
    conc_layers = layers.concatenate([small_layer_up,large_ip_layer])
    # TODO Add some number of separable convolution layers
    output_layer = separable_conv2d_batchnorm(conc_layers, filters)
    return output_layer

``` 

The implementation of final model is featured below:


``` python
def fcn_model(inputs, num_classes):
    
    # TODO Add Encoder Blocks. 
    # Remember that with each encoder layer, the depth of your model (the number of filters) increases.
    extra_layer = conv2d_batchnorm(inputs, 16, 1, 1)
    encoder1 = encoder_block(extra_layer, 32, 2)             # input, #filters, #strides , note default kernel_size=3
    encoder2 = encoder_block(encoder1, 64, 2)
    encoder3 = encoder_block(encoder2, 128, 2)
    
    # TODO Add 1x1 Convolution layer using conv2d_batchnorm().
    conv_layer = conv2d_batchnorm(encoder3, 256, 1, 1)
    
    # TODO: Add the same number of Decoder Blocks as the number of Encoder Blocks
    decoder1 = decoder_block(conv_layer, encoder2, 128)     # note default kernel_size=3
    decoder2 = decoder_block(decoder1, encoder1, 64) 
    decoder3 = decoder_block(decoder2, extra_layer, 32) 
    
    x = decoder3
    
    # The function returns the output layer of your model. "x" is the final layer obtained from the last decoder_block()
    return layers.Conv2D(num_classes, 1, activation='softmax', padding='same')(x)


``` 


Where the extra_layer, and 1 x 1 convolution were implemented as regular convolutions, and both regular and separable convolutions include batch normalization and a ReLU activation function. 




## Results

The previous network was trained (with the Adam optimizer) and validated using Udacity's image data set. The final model can be found in the *data/weights/* directory, as *model_weights5*.

As shown below, the resulting training and validation curves are almost flat, with an average loss of around 0.027 for the training curve, and of 0.010 for the validation curve. This means that right after the first epoch has passed (with its 200 batches), the network is already able to reduce the error significantly.

![alt text][image_02]

After training, the final network architecture was tested in the quadcopter simulator, resulting in a *final score* of 0.46 (above the *base requirement* of 0.4). This allowed the quadcopter to follow the "hero" throughout crowded and uncrowded areas in a virtual environment.

The network performed particularly well once the "hero" was identified, which was reflected in an *average intersection over union* of 0.92.
The following images show the quadcopter following the hero through a crowd:

![alt text][image_03]

![alt text][image_cover]

Click [here](https://www.youtube.com/watch?v=6PC8Qb1PBAs) to see a video of this simulation.



## Limitations

* The network does not perform so well when the target is far away, this indicates that the provided data set lacks images taken at a distance from different angles.

* The network was trained to identify a specific human character with specific characteristics. If these characteristics were modified or if a new characater was used as hero (another human or animal), then the network would have to be trained and tuned again. Moreover, a new data set would be needed containing the new "hero" character to follow.

## Future Enhancements

* Only Udacity's provided data set was used for training. Gathering more data containing the "hero" from different distances and angles could increase the final score.

* The final network architecture accomplished this project's goals and doesn't take alot of time to train. However, I would like to experiment with even more complex architectures and analyze their impact on the final score.

* Finally, tuning the hyperparameters by hand can be inefficient and time consuming. A solution could be to use the grid search capability from the scikit-learn python machine learning library to automate this process, as shown [here](https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/).





