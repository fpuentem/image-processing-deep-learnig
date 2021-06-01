#gera2.py - Visualizar data augmentation - testado em Colab com TF2
#Programa para ver imagem original do CIFAR-10
#juntamente com a obtida pelo data augmentation
import os; os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow.keras as keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from inspect import currentframe, getframeinfo
from matplotlib import pyplot as plt
import numpy as np; import sys; import os


nl, nc = 32,32; input_shape = (nl, nc, 3)
(ax, ay), (qx, qy) = cifar10.load_data()
ax = ax.astype('float32'); ax /= 255 #0 a 1
qx = qx.astype('float32'); qx /= 255 #0 a 1
datagen1 = ImageDataGenerator() #Nao distorce imagem
datagen2 = ImageDataGenerator( #Distorce imagem
#width_shift_range=0.1, # randomly shift images horizontally - Fraction of width
rotation_range = 45,
#height_shift_range=0.1, # randomly shift images vertically - Fraction of height
fill_mode='nearest', # Preenche pixels fora do dominio com valores dentro da imagem
#horizontal_flip=True
) # Espelha imagem horizontalmente

#Compute quantities required for featurewise normalization
#(std, mean, and principal components if ZCA whitening is applied).
#Caso nao use, nao precisa destes comandos.

datagen1.fit(ax); datagen2.fit(ax)
batch_size = 10

#Importante usar a mesma semente, para pegar as mesmas imagens
augdata1=datagen1.flow(ax, batch_size=batch_size, seed=7)
augdata2=datagen2.flow(ax, batch_size=batch_size, seed=7)
a1=augdata1.next(); a2=augdata2.next()
for i in range(3):
    fig, eixo = plt.subplots(1, 2)
    eixo[0].imshow(a1[i]); eixo[0].axis("off")
    eixo[1].imshow(a2[i]); eixo[1].axis("off")
    plt.show()