# Short hands-on introduction to deep learning, with Keras.

## Installing Keras

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

TODO Check if pip version is compatible with our examples.

## Running the handwritten digit recognition code

```python mnist_mlp.py```

The results are written to standard output, and a confusion matrix
visualization is written to ```vis.png```.

## Running the deep dream code

You need the pre-trained VGG16 model weights file called ```vgg16_weights.h5```, from
https://drive.google.com/file/d/0Bz7KyqmuGsilT0J5dmRCM0ROVHc/view
You have to open the link in a browser, wget is circumvented.

```python vgg16_deep_dream.py input_image.png output_file_prefix```

The final output is in ```output_file_prefix_at_iteration_9.png```.
