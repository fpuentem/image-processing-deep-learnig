# mlp1.py
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow.keras as keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras import optimizers
import numpy as np
import sys
(AX, AY),(QX, QY) = mnist.load_data()
AX = 255 - AX
QX = 255 - QX

nclasses =10
AY2 = keras.utils.to_categorical( AY, nclasses )
QY2 = keras.utils.to_categorical( QY, nclasses )
nl, nc = AX.shape[1], AX.shape[2] # 28, 28
AX = AX.astype('float32')/255.0 # 0 a 1
QX = QX.astype('float32')/255.0 # 0 a 1

model = Sequential()
model.add( Flatten( input_shape=(nl,nc) ) )
model.add( Dense(400, activation='sigmoid') )
model.add( Dense(nclasses, activation='sigmoid') )
# from keras.utils import plot_model
# plot_model(model, to_file='mlp1.png', show_shapes=True)

model.summary() # opt=optimizers.sgd(lr=0.5)
opt = optimizers.Adam()

model.compile( optimizer=opt,              
               loss='mse',
               # loss='categorical_crossentropy',              
               metrics=['accuracy'] )

model.fit(  AX, AY2,          
            batch_size=100,          
            epochs=40,          
            verbose=True )

score = model.evaluate( QX, QY2, verbose=False )
print('Test loss: ', score[0])
print('Test accuracy: ', score[1])
model.save('mlp1.h5')