'''Trains a simple deep NN on the MNIST dataset.

Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

import numpy as np
from collections import defaultdict
import scipy.misc

np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils


batch_size = 128
nb_classes = 10
image_size = 28
nb_features = image_size * image_size
nb_epoch = 20

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

layer_size = 100
dropout_rate = 0.2
nonlinearity = 'tanh'
inputs = Input(shape=(nb_features,))
net = Dense(layer_size, activation=nonlinearity)(inputs)
# net = Dropout(dropout_rate)(net)
# net = Dense(layer_size, activation=nonlinearity)(net)
# net = Dropout(dropout_rate)(net)
predictions = Dense(nb_classes, activation='softmax')(net)

model = Model(input=inputs, output=predictions)

model.summary()

# That was it, neural network is created now.
# We compile it with the given loss and optimizer:

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

# Let's train it!

history = model.fit(X_train, Y_train,
                    batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=1, validation_data=(X_test, Y_test))

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
