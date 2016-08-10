# Short hands-on introduction to deep learning, with Keras.

## Installing Keras

**UPDATE:** Check out the cool install script by Akos Hochrein:
https://github.com/akoskaaa/hands-on-deep-learning

Installing Theano and its dependencies is the hard part:
http://deeplearning.net/software/theano/install.html#install

After that, a simple
```sudo pip install keras```
is enough, although the bleeding edge git version is often better than the pip version.

If you have an Nvidia video card, you are advised to also install CUDA and CuDNN.
The first steps of the Theano installation instructions explain how:

For Ubuntu:
http://deeplearning.net/software/theano/install_ubuntu.html#install-ubuntu

For OS X:
http://deeplearning.net/software/theano/install.html#gpu-macos

The examples are 5-15 times faster when running on GPU.


TODO Check if pip version is compatible with our examples.

TODO Set up ```.theanorc```.

## Running the handwritten digit recognition code

```python mnist_mlp.py```

The results are written to standard output, and a confusion matrix
visualization is written to ```vis.png```.

## Running the VGG-16 image classification code

Unlike the handwritten digit, now we don't train a network, we just use one
pre-trained by smart people.

You need the pre-trained VGG16 model weights file called ```vgg16_weights.h5```, from
https://drive.google.com/file/d/0Bz7KyqmuGsilT0J5dmRCM0ROVHc/view
You have to open the link in a browser, wget is circumvented.

```python vgg16_predictor.py image1.png image2.png ...```

Output looks like this:

```
jersey = 0.194810 , electric_guitar = 0.056235 , acoustic_guitar = 0.039047 , lab_coat = 0.037690 , drumstick = 0.036288 ,
lorikeet = 0.937200 , hummingbird = 0.022311 , peacock = 0.013273 , macaw = 0.011136 , jacamar = 0.003892 ,
goldfish = 0.164630 , axolotl = 0.083203 , common_newt = 0.062816 , starfish = 0.057640 , eel = 0.038394 ,
streetcar = 0.330774 , palace = 0.210260 , trolleybus = 0.080859 , cinema = 0.044910 , traffic_light = 0.032249 ,
```

## Running the deep dream code

This code also requires the pre-trained VGG16 model weights, see above.

```python vgg16_deep_dream.py input_image.png output_file_prefix```

The final output is in ```output_file_prefix_at_iteration_9.png```.
