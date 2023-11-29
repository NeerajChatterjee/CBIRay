from keras.applications.vgg16 import VGG16
from keras import layers
from keras import models


def create_vgg_model():
    conv_base = VGG16(weights='imagenet',
                      include_top=False,
                      input_shape=(240, 240, 3))

    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(128, activation='relu'))

    model.load_weights('./backend/cnn/vgg_weights.h5')

    return model
