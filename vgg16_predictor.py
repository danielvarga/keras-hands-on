import sys
import json
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import scipy.misc, numpy as np

from vgg16 import *


if __name__ == "__main__":
    print "constructing network..."
    model = VGG_16('vgg16_weights.h5')
    print "done"
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')

    class_names = json.load(open("imagenet_class_index.json"))

    for filename in sys.argv[1:]:
        im = scipy.misc.imresize(scipy.misc.imread(filename), (224, 224)).astype(np.float32)
        im = im[:, :, :3]
        im[:,:,0] -= 103.939
        im[:,:,1] -= 116.779
        im[:,:,2] -= 123.68
        im = im.transpose((2,0,1))
        im = np.expand_dims(im, axis=0)

        out = model.predict(im).flatten()
        prediction = np.argmax(out)
        top5 = sorted(zip(out, range(len(out))), reverse=True)[:5] # (probability, class_id) pairs.
        top5_probs_and_names = ["%s = %f ," % (class_names[str(prediction)][1], probability) for (probability, prediction) in top5]
        top5_names = [class_names[str(prediction)][1] for (probability, prediction) in top5]

        print " ".join(top5_probs_and_names).encode("utf-8")
