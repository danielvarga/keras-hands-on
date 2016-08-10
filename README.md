# Short hands-on introduction to deep learning, with Keras.

## Installing Keras

The best way is to use [Akos Hochrein's](https://github.com/akoskaaa) installation script
```setup.sh```, now part of this repo (thanks, Akos!).
It can work with either the Theano or the Tensorflow backend, and it installs into a virtualenv.

With [Theano](http://deeplearning.net/software/theano/)
```
./setup.sh
```

With [TensorFlow](https://www.tensorflow.org/)
```
./setup.sh --tensorflow
```

If you have an Nvidia video card, you are advised to install CUDA and CuDNN.
The first steps of the Theano installation instructions explain how:

For Ubuntu:
http://deeplearning.net/software/theano/install_ubuntu.html#install-ubuntu

For OS X:
http://deeplearning.net/software/theano/install.html#gpu-macos

The examples are 5-15 times faster when running on a GPU.
On a Linux machine, it's also worth looking into the linear algebra libraries installed,
see the above Ubuntu link for more information (Chapter: Troubleshooting: Make sure you have a BLAS library).


## Introduction to tensor computation

The ```theano``` subdirectory contains code snippets illustrating some tensor library concepts.
For (much) more information, you can check out http://deeplearning.net/software/theano/tutorial/
or https://github.com/nlintz/TensorFlow-Tutorials .


## Running the handwritten digit recognition code

We train and evaluate our own handwritten digit recognition network.

```python mnist_mlp.py```

The evaluation results are written to standard output, and a confusion matrix
visualization is written to ```vis.png```. The confusion matrix looks like this:

![MNIST confusion matrix](http://people.mokk.bme.hu/~daniel/keras-hands-on-mnist-confusion.png "MNIST confusion matrix")


## Running the VGG-16 image classification code

Unlike with the handwritten digit example, now we don't train a network, we just use one
designed and trained by smart people.

You need the pre-trained VGG16 model weights file called ```vgg16_weights.h5```, from
https://drive.google.com/file/d/0Bz7KyqmuGsilT0J5dmRCM0ROVHc/view
You have to open the link in a browser, wget is circumvented.

```python vgg16_predictor.py image1.png image2.png ...```

Output looks like this. The numbers are TOP 5 prediction probabilities, each line is corresponding to an image:

```
jersey = 0.194810 , electric_guitar = 0.056235 , acoustic_guitar = 0.039047 , lab_coat = 0.037690 , drumstick = 0.036288 ,
lorikeet = 0.937200 , hummingbird = 0.022311 , peacock = 0.013273 , macaw = 0.011136 , jacamar = 0.003892 ,
streetcar = 0.330774 , palace = 0.210260 , trolleybus = 0.080859 , cinema = 0.044910 , traffic_light = 0.032249 ,
```

By the way, most of the ```vgg16_predictor.py``` code is obsoleted by
the recent publication of https://github.com/fchollet/deep-learning-models ,
which is more robust and general.


## Running the deep dream code

This code also requires the pre-trained VGG16 model weights, see above.

```python vgg16_deep_dream.py input_image.png output_file_prefix```

The final output is in ```output_file_prefix_at_iteration_9.png```.
