######################################################
### CREDITS to: https://github.com/serengil/deepface
######################################################

import os
import gdown
from basemodels import Facenet


def loadModel(
    url="https://github.com/serengil/deepface_models/releases/download/v1.0/facenet512_weights.h5",
):

    model = Facenet.InceptionResNetV2(dimension=512)

    home = (os.path.join(os.path.expanduser('~')))

    if os.path.isfile(home + "/.deepface/weights/facenet512_weights.h5") != True:
        print("facenet512_weights.h5 will be downloaded...")

        output = home + "/.deepface/weights/facenet512_weights.h5"
        gdown.download(url, output, quiet=False)

    model.load_weights(home + "/.deepface/weights/facenet512_weights.h5")

    return model
