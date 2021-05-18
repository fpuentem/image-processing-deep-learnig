#regression.py - pos2021
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow.keras as keras
from keras.models import Sequential 
from keras.layers import Dense, Activation 
from keras import optimizers
import numpy as np # Modelo de rede
model = Sequential()
model.add( Dense( 2, activation='sigmoid', input_dim=2 ))
model.add( Dense( 2, activation='sigmoid' ))
sgd = optimizers.SGD( lr=1 )
model.compile( optimizer=sgd,loss='mse' )# Gera dado artificial
AX = np.matrix( '0.9 0.1; 0.1 0.9', dtype='float32' )
AY = np.matrix( '0.1 0.9; 0.9 0.1', dtype='float32' )
# As alternativas sao batch_size 2 ou 1
model.fit( AX, AY, epochs=1000, batch_size=2, verbose=False )
QX = np.matrix( '0.9 0.1; 0.1 0.9; 0.8 0.0; 0.2 0.9',dtype='float32' )
print( QX )
QP = model.predict(QX)
print( QP )