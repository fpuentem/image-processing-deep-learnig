# mnist1.py
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow.keras as keras
from keras.datasets import mnist
from matplotlib import pyplot as plt

(AX, AY),(QX, QY) = mnist.load_data()

print("AX:", AX.shape, AX.dtype)
print("AY:", AY.shape, AY.dtype)
print("QX:", QX.shape, QX.dtype)
print("QY:", QY.shape, QY.dtype)

AX = 255 - AX
QX = 255 - QX

plt.imshow(AX[0],cmap="gray",interpolation="nearest")
plt.show()

nclasses = 10

AY2 = keras.utils.to_categorical(AY, nclasses)
QY2 = keras.utils.to_categorical(QY, nclasses)

print("AY[0]:", AY[0])
print("AY2[0]:", AY2[0])
print("AY2:", AY2.shape, AY2.dtype)