from keras.applications.vgg16 import VGG16
from keras import layers
from keras import models
from keras.applications.densenet import DenseNet121
from keras.applications.inception_v3 import InceptionV3
from keras.models import Sequential
from keras.layers import Dense, Dropout


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

    model.load_weights('./cnn/vgg_weights.h5')

    return model


def create_densenet121_model():
    base_model = DenseNet121(input_shape=(224, 224, 3), include_top=False, weights='imagenet', classes=3, pooling='avg')

    model = Sequential()
    model.add(base_model)
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))

    model.load_weights('./densenet121/densenet121_weights.h5')

    return model


def create_inception_model():
    base_model = InceptionV3(include_top=False, weights="imagenet", input_shape=(224, 224, 3), pooling='avg', classes=3)

    model = Sequential()
    model.add(base_model)
    model.add(Dense(512, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))

    model.load_weights('./inception/inception_weights.h5')

    return model
