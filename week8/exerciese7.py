# pred1.py
import os 
os.environ[ 'TF_CPP_MIN_LOG_LEVEL' ] = '3'
import tensorflow.keras as keras 
from keras.datasets import mnist
from keras.models import load_model
from keras.utils import to_categorical
# import os; os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
(_, _), (QX, QY) = mnist.load_data()

QX = 255 - QX
nclasses = 10

QY2 = keras.utils.to_categorical( QY, nclasses )
nl, nc = QX.shape[1], QX.shape[2] # 28, 28
QX = QX.astype( 'float32' )/255.0 # 0 a 1

model = load_model( 'mlp2.h5' )
score = model.evaluate( QX, QY2, verbose=False )

print( 'Test loss: ', score[0] )
print( 'Test accuracy: ', score[1] )

QP = model.predict( QX )
QP = QP.argmax( axis=-1 )