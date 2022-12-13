
from keras_vggface.vggface import VGGFace

#model = VGGFace(model='vgg16')
# same as the following
#model = VGGFace() # vgg16 as default
model = VGGFace(model='resnet50')
#model = VGGFace(model='senet50')


import numpy as np
from tensorflow import keras
import keras.utils as image
from keras_vggface.vggface import VGGFace
from keras_vggface import utils

# load the image
img = image.load_img(
    r"c:\Users\DIAC\Desktop\dl\faceapp\Matthias_Sammer.png",
    target_size=(224, 224))

# prepare the image
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = utils.preprocess_input(x, version=1)

# perform prediction
preds = model.predict(x)
print('Predicted:', utils.decode_predictions(preds))