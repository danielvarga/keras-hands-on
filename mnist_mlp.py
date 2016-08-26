'''Trains a simple neural network on the MNIST dataset.
'''

import numpy as np
from collections import defaultdict
import scipy.misc

np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras.layers.advanced_activations import PReLU
from keras.callbacks import Callback
import keras.backend as K

batch_size = 128
nb_classes = 10
image_size = 28
nb_features = image_size * image_size
nb_epoch = 20

# Is set in add_prelu_relaxation_loss()
prelu_target_alpha = None

# The data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# flatten the 28x28 images to arrays of length 28*28:
X_train = X_train.reshape(60000, nb_features)
X_test = X_test.reshape(10000, nb_features)

# convert brightness values from bytes to floats between 0 and 1:
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices:
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

# We build the neural network:

layer_size = 512
dropout_rate = 0.2
inputs = Input(shape=(nb_features,))

initial_alpha = 1.0 / 3

def const_initializer(shape, name=None):
    # return K.ones(shape, name=name) * initial_alpha
    return K.variable(initial_alpha * np.ones(shape), name=name)

net = Dense(layer_size)(inputs)
net = PReLU(init=const_initializer)(net)
net = Dropout(dropout_rate)(net)
net = Dense(layer_size)(net)
net = PReLU(init=const_initializer)(net)
net = Dropout(dropout_rate)(net)
predictions = Dense(nb_classes, activation='softmax')(net)

model = Model(input=inputs, output=predictions)

def prelu_layers(model):
    ls = model.layers
    ls = [l for l in ls if l.name.startswith('prelu')]
    return ls

model.summary()

# That was it, neural network is created now.
# We compile it with the given loss and optimizer:


def pretty_histogram(w):
    c, s = np.histogram(w)
    print
    print "\t".join(map(lambda num: "%.3f" % num, s))
    print "\t".join(map(str, c))

class WeightDump(Callback):
    def on_epoch_end(self, epoch, logs={}):
        ls = prelu_layers(self.model)
        for l in ls:
            w = l.get_weights()
            pretty_histogram(w)

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])


prelu_relaxation_weight = 1.0

def add_prelu_relaxation_loss(model):
    global prelu_target_alpha
    prelu_target_alpha = K.variable(1.0/3)
    for layer in prelu_layers(model):
        layer_loss = K.mean((layer.alphas - prelu_target_alpha) ** 2)
        model.total_loss += layer_loss * prelu_relaxation_weight

add_prelu_relaxation_loss(model)

class PReLUAlphaDecay(Callback):
    def __init__(self, start, end):
        self.decrement = (start-end) / nb_epoch
        K.set_value(prelu_target_alpha, start)
    def on_epoch_end(self, epoch, logs={}):
        new_prelu_target_alpha = K.get_value(prelu_target_alpha) - self.decrement
        K.set_value(prelu_target_alpha, new_prelu_target_alpha)
        print "prelu_target_alpha", new_prelu_target_alpha

# Let's train it!

history = model.fit(X_train, Y_train,
                    batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=1, validation_data=(X_test, Y_test),
                    callbacks=[WeightDump(), PReLUAlphaDecay(1.0, 0.0)])

score = model.evaluate(X_test, Y_test, verbose=0)
print 'Test score:', score[0]
print 'Test accuracy:', score[1]

Y_pred = model.predict(X_test, batch_size=batch_size)
y_pred = np.argmax(Y_pred, axis=1)
assert float(np.sum(y_test==y_pred)) / len(y_test) == score[1]

# Visualizing the classification:

bucket_size = 10
vis_image_size = nb_classes * image_size * bucket_size
vis_image = 255 * np.ones((vis_image_size, vis_image_size), dtype='uint8')
example_counts = defaultdict(int)
for (predicted_tag, actual_tag, image) in zip(y_pred, y_test, X_test):
    image = ((1 - image) * 255).reshape((image_size, image_size)).astype('uint8')
    example_count = example_counts[(predicted_tag, actual_tag)]
    if example_count >= bucket_size**2:
        continue
    tilepos_x = bucket_size * predicted_tag
    tilepos_y = bucket_size * actual_tag
    tilepos_x += example_count % bucket_size
    tilepos_y += example_count // bucket_size
    pos_x, pos_y = tilepos_x * image_size, tilepos_y * image_size
    vis_image[pos_y:pos_y+image_size, pos_x:pos_x+image_size] = image
    example_counts[(predicted_tag, actual_tag)] += 1

vis_image[::image_size * bucket_size, :] = 0
vis_image[:, ::image_size * bucket_size] = 0
scipy.misc.imsave("vis.png", vis_image)
