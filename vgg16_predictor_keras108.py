import sys

from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions
import scipy.misc, numpy as np


if __name__ == "__main__":
    print "constructing network..."
    model = VGG16(weights='imagenet', include_top=True)
    print "done"

    # Forced initialization of keras.applications.imagenet_utils.CLASS_INDEX
    # imagenet_utils kind of hides the CLASS_INDEX from us, that's why this hackery is necessary.
    _ = decode_predictions(np.zeros((1, 1000)))
    from keras.applications.imagenet_utils import CLASS_INDEX

    for filename in sys.argv[1:]:
        img = image.load_img(filename, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        # preds = model.predict(x)
        # print 'Predicted:', decode_predictions(preds)[0][1]

        out = model.predict(x).flatten()
        prediction = np.argmax(out)
        top5 = sorted(zip(out, range(len(out))), reverse=True)[:5] # (probability, class_id) pairs.
        top5_probs_and_names = ["%s = %f ," % (CLASS_INDEX[str(prediction)][1], probability) for (probability, prediction) in top5]
        top5_names = [CLASS_INDEX[str(prediction)][1] for (probability, prediction) in top5]
        print " ".join(top5_probs_and_names).encode("utf-8")
