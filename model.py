import tensorflow as tf
from keras.applications import VGG16
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os
train_data_dir = 'drive/MyDrive/DL/gestures/train'
validation_data_dir = 'drive/MyDrive/DL/gestures/test'
img_width, img_height = 224, 224
batch_size = 32
epochs = 5
num_classes = 10
train_data = ImageDataGenerator(
    rescale=1.0 / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
validation_data = ImageDataGenerator(rescale=1.0 / 255)
base = VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))
for layer in base.layers:
    layer.trainable = False
model = Sequential()
model.add(base)
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.compile(optimizer=Adam(learning_rate=0.00001), loss='categorical_crossentropy', metrics=['accuracy'])
train_generator = train_data.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)
validation_generator = validation_data.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

graphs=model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=len(validation_generator)
)
model.save('model.h5')