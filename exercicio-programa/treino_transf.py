# treino_transf.py
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH']='true'

# Instantiating the VGG16 convolutional base
from tokenize import PlainToken
from keras.applications import VGG16
from tensorflow.python.ops.variables import trainable_variables

import matplotlib.pyplot as plt

from keras import models
from keras import layers

from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers

conv_base = VGG16(weights='imagenet', 
                  include_top=False, 
                  input_shape=(150, 150, 3))

conv_base.summary()

# Adding a densely connected classifier on the top of the convolutional base
model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()

# Freesing all layers up to a specific one
conv_base.trainable = True

set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

# Fine-tuning the model
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_dir = '/home/fabricio/projects/usp-projects/image-processing-deep-learnig/cats_and_dogs_small/train'
test_dir = '/home/fabricio/projects/usp-projects/image-processing-deep-learnig/cats_and_dogs_small/test'
validation_dir = '/home/fabricio/projects/usp-projects/image-processing-deep-learnig/cats_and_dogs_small/validation'

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(150, 150),
                                                    batch_size=32,
                                                    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(validation_dir,
                                                        target_size=(150, 150),
                                                        batch_size=32,
                                                        class_mode='binary')

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-5), 
              metrics=['accuracy'])
            
history = model.fit_generator(
                    train_generator, 
                    steps_per_epoch=100,
                    epochs=45,
                    validation_data=validation_generator,
                    validation_steps=50)

# Saving the model
model.save('./transf.h5')
            
# Smoothing the plots

# ['accuracy', 'loss', 'val_accuracy', 'val_loss']
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

def smooth_curve(points, factor=0.8):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

plt.plot(epochs, smooth_curve(acc), 'bo', label='Smoothed training acc')
plt.plot(epochs, smooth_curve(val_acc), 'b', label='smoothed validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.savefig("training_tranf_1.png")

plt.figure()

plt.plot(epochs, smooth_curve(loss), 'bo', label='Smoothed training loss')
plt.plot(epochs, smooth_curve(val_loss), 'b', label='Smoothed validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig("training_tranf_2.png")

plt.show()
