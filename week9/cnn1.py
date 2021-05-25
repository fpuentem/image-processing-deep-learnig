#cnn1.py - pos2021
import os
from tensorflow.python.keras import regularizers
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
# Se acontecer erro "Failed to get convolution algorithm” coloque o comando abaixo 
# https://github.com/tensorflow/tensorflow/issues/43174
# https://stackoverflow.com/questions/53698035/failed-to-get-convolution-algorithm-this-is-probably-because-cudnn-failed-to-in
os.environ['TF_FORCE_GPU_ALLOW_GROWTH']='true'
import tensorflow.keras as keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dropout, Conv2D, MaxPooling2D, Dense, Flatten
from keras import optimizers
import numpy as np
import sys
import os
from time import time

(AX, AY),(QX, QY) = mnist.load_data()# AX [60000,28,28] AY [60000,]
AX = 255 - AX
QX = 255 - QX
nclasses = 10
AY2 = keras.utils.to_categorical( AY, nclasses )# 3 -> 0001000000
QY2 = keras.utils.to_categorical( QY, nclasses )
nl, nc = AX.shape[1], AX.shape[2]#28, 28

AX = AX.astype( 'float32' )/255.0# 0 a 1
QX = QX.astype( 'float32' )/255.0# 0 a 1
AX = np.expand_dims(AX,axis=3)# AX [60000,28,28,1]
QX = np.expand_dims(QX,axis=3)

AX = AX[:6000, :, :, :]
AY2 = AY2[:6000]
print("***** Dataset  Size *****")
print(">> Shape of AX : {}".format( AX.shape ))
print(">> Shape of AY2 : {}".format( AY2.shape ) )


model = Sequential()# 28x28
model.add( Conv2D(  20, kernel_size=(5,5), activation='relu',  # 20
                    input_shape=(nl, nc,1)) )# 24x24x20
model.add(MaxPooling2D(pool_size=(2,2))) # 12x12x20
model.add(Conv2D( 40, kernel_size=(5,5), activation='relu' )) # 8x8x40
model.add(MaxPooling2D(pool_size=(2,2))) # 4x4x40
model.add(Flatten()) # 640
model.add(Dense(500, activation='relu')) # 1000
model.add(Dense(nclasses, activation='softmax')) # 10

from keras.utils import plot_model
plot_model(model, to_file='cnn1.png', show_shapes=True)
model.summary()

opt = optimizers.Adam()
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

t0 = time()
model.fit(AX, AY2, batch_size=100, epochs=30, verbose=2) # batch_size=100
t1 = time()
print("Tempo de treino: %.2f s"%(t1-t0))

score = model.evaluate(QX, QY2, verbose=False)
print("***** Time Result *****")
print('>> Test loss: %.4f'%(score[0]))
print('>> Test accuracy: %.2f %%'%(100*score[1]))
print('>> Test error: %.2f %%'%(100*(1-score[1])))

model.save('cnn1.h5')
