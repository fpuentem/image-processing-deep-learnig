# vgg3.py 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH']='true'
import tensorflow.keras as keras
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras import backend as K
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dropout, Conv2D, MaxPooling2D, Dense, Flatten, BatchNormalization
from keras import optimizers
import matplotlib.pyplot as plt;
import numpy as np
import cv2

def impHistoria(history):
    print(history.history.keys())  
    plt.plot(history.history['accuracy']); 
    plt.plot(history.history['val_accuracy'])  
    plt.title('model accuracy'); 
    plt.ylabel('accuracy'); 
    plt.xlabel('epoch')  
    plt.legend(['train','test'], loc='upper left'); 
    plt.show()  
    plt.plot(history.history['loss']); 
    plt.plot(history.history['val_loss'])  
    plt.title('model loss'); 
    plt.ylabel('loss'); 
    plt.xlabel('epoch')  
    plt.legend(['train','test'], loc='upper left'); 
    plt.show()

batch_size = 100 
num_classes = 10 
epochs = 35
nl, nc = 32,32 
input_shape =(nl, nc, 1)
# Load data of cifar10
# (ax, ay),(qx, qy)= cifar10.load_data()
# grab the Fashion MNIST dataset (if this is your first time running
# this the dataset will be automatically downloaded)
print("[INFO] loading Fashion MNIST...")
((ax, ay), (qx, qy)) = fashion_mnist.load_data()
# if we are using "channels first" ordering, then reshape the design
# matrix such that the matrix is:
# 	num_samples x depth x rows x columns
if K.image_data_format() == "channels_first":
	ax = ax.reshape((ax.shape[0], 1, 28, 28))
	qx = qx.reshape((qx.shape[0], 1, 28, 28))
 
# otherwise, we are using "channels last" ordering, so the design
# matrix shape should be: num_samples x rows x columns x depth
else:
	ax = ax.reshape((ax.shape[0], 28, 28, 1))
	qx = qx.reshape((qx.shape[0], 28, 28, 1))

ax = ax.astype('float32') 
ax /= 255 # 0 a 1
qx = qx.astype('float32') 
qx /= 255 # 0 a 1
ay = keras.utils.to_categorical(ay, num_classes)
qy = keras.utils.to_categorical(qy, num_classes)

print("Shape ax: {}".format(ax.shape))
print("Shape qx: {}".format(qx.shape))
print("Shape ay: {}".format(ay.shape))
print("Shape qy: {}".format(qy.shape))
print(ax.shape[0])

AX = np.zeros((60000, 32, 32, 1))
for i in range(ax.shape[0]):
    AX[i, :, :, 0] = cv2.copyMakeBorder(ax[i,:,:,0],2,2,2,2,cv2.BORDER_CONSTANT,255)
print("Shape AX: {}".format(AX.shape))

QX = np.zeros((10000, 32, 32, 1))
for i in range(qx.shape[0]):
    QX[i, :, :, 0] = cv2.copyMakeBorder(qx[i,:,:,0],2,2,2,2,cv2.BORDER_CONSTANT,255)
print("Shape QX: {}".format(QX.shape))

model = Sequential()
model.add(Conv2D(64, kernel_size=(3,3), activation='relu', padding='same', input_shape=input_shape))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Conv2D(64, kernel_size=(3,3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2))) # 20x16x16x3
model.add(Conv2D(128, kernel_size=(3,3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Conv2D(128, kernel_size=(3,3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2))) # 40x8x8x3
model.add(Conv2D(256, kernel_size=(3,3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Conv2D(256, kernel_size=(3,3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))# 80x4x4x3
model.add(Conv2D(512, kernel_size=(3,3), activation='relu', padding='same'))# 160x4x4x3
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Conv2D(512, kernel_size=(3,3), activation='relu', padding='same'))# 160x4x4x3
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Conv2D(512, kernel_size=(3,3), activation='relu', padding='same'))# 160x4x4x3
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))# 160x2x2x3
model.add(Conv2D(512, kernel_size=(3,3), activation='relu', padding='same'))# 160x2x2x3
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Conv2D(512, kernel_size=(3,3), activation='relu', padding='same'))# 160x2x2x3
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Conv2D(512, kernel_size=(3,3), activation='relu', padding='same'))# 160x2x2x3
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))# 160x1x1x3
model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(num_classes,activation='softmax'))

from keras.utils import plot_model
plot_model(model, to_file='vgg3.png', show_shapes=True); 
model.summary()
datagen = ImageDataGenerator(   featurewise_center=False,# set input mean to 0 over the dataset  
                                samplewise_center=False,# set each sample mean to 0  
                                featurewise_std_normalization=False,# divide inputs by std of the dataset  
                                samplewise_std_normalization=False,# divide each input by its std  
                                zca_whitening=False,# apply ZCA whitening  
                                rotation_range=15,# randomly rotate images in the range (degrees, 0 to 180)  
                                width_shift_range=0.1,# randomly shift images horizontally (fraction of total width)  
                                height_shift_range=0.1,# randomly shift images vertically (fraction of total height)  
                                horizontal_flip=True,# randomly flip images  
                                vertical_flip=True)# randomly flip images

datagen.fit(AX)
opt = optimizers.Adam()
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(datagen.flow(AX,ay,batch_size=batch_size), steps_per_epoch=AX.shape[0]//batch_size,          
                    epochs=epochs, verbose=2, validation_data=(QX, qy))
impHistoria(history)
score = model.evaluate(QX, qy, verbose=0)
print('*************')
print('>> Test loss:', score[0])
print('>> Test accuracy:', score[1])
model.save('vgg3.h5')