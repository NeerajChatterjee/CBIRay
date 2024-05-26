from keras.applications.inception_v3 import InceptionV3
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import (
    Dense, Dropout
)

train_dir = "../static/Data/train"
test_dir = "../static/Data/test"

# image_preprocessing():
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    samplewise_center=True,
    samplewise_std_normalization=True
)

train = image_generator.flow_from_directory(train_dir,
                                            batch_size=8,
                                            shuffle=True,
                                            class_mode='categorical',
                                            target_size=(224, 224))

test = image_generator.flow_from_directory(test_dir,
                                           batch_size=1,
                                           shuffle=False,
                                           class_mode='categorical',
                                           target_size=(224, 224))


num_pneumonia = len(os.listdir(os.path.join(train_dir, 'PNEUMONIA')))
num_normal = len(os.listdir(os.path.join(train_dir, 'NORMAL')))
num_covid19 = len(os.listdir(os.path.join(train_dir, 'COVID19')))

# Class weights

weight_for_0 = num_pneumonia / (num_normal + num_pneumonia + num_covid19)
weight_for_1 = num_normal / (num_normal + num_pneumonia + num_covid19)
weight_for_2 = num_covid19 / (num_normal + num_pneumonia + num_covid19)

class_weight = {0: weight_for_0, 1: weight_for_1, 2: weight_for_2}

print(f"Weight for class 0: {weight_for_0:.2f}")
print(f"Weight for class 1: {weight_for_1:.2f}")
print(f"Weight for class 2: {weight_for_2:.2f}")

base_model = InceptionV3(include_top=False, weights="imagenet", input_shape=(224, 224, 3), pooling='avg', classes=3)

model = Sequential()
model.add(base_model)
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dense(3, activation='sigmoid'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

r = model.fit(
    train,
    epochs=25,
    validation_data=test,
    class_weight=class_weight,
    steps_per_epoch=100,
    validation_steps=25,
)

plt.xlabel("Epochs")
plt.ylabel("Percentage (%)")
plt.title("InceptionV3 model training metrics")

r.history['accuracy'] = [x*100 for x in r.history['accuracy']]
r.history['loss'] = [x*100 for x in r.history['loss']]

plt.plot([i for i in range(1, 26)], r.history['accuracy'], color='b', label='Accuracy', marker='o')
plt.plot([i for i in range(1, 26)], r.history['loss'], color='r', label='Loss')
print(r.history['accuracy'])
print(r.history['loss'])
plt.axis([0, 30, 0, 100])

plt.legend()
plt.show()

# plt.plot(r.history['accuracy'])
# plt.title('Model Accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.show()
#
# plt.plot(r.history['loss'])
# plt.title('Model Loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.show()

# model2 = Model(model.input, model.layers[-2].output)
#
# model2.save_weights('inception_weights.h5')