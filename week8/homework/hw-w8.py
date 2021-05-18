from keras.utils import plot_model
from keras.datasets import fashion_mnist
import sys
from matplotlib import pyplot as plt
import numpy as np
from keras import optimizers
from keras.layers import MaxPooling2D
from keras.layers import Conv2D
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.datasets import mnist
import tensorflow.keras as keras
import cv2
import os
from re import DEBUG
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


NUM_EPOCHS = 45
DEBUG = 0
# Load dataset
(AX, AY), (QX, QY) = fashion_mnist.load_data()
AX = 255 - AX
QX = 255 - QX

# Categories of dataset
categories = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
              "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Shape and data type of dataset
if DEBUG:
    print("> AX: \n shape: {}, dtype:{}", AX.shape, AX.dtype)
    print("> AY: \n shape: {}, dtype:{}", AY.shape, AY.dtype)
    print("> QX: \n shape: {}, dtype:{}", QX.shape, QX.dtype)
    print("> QY: \n shape: {}, dtype:{}", QY.shape, QY.dtype)

    print("> Categories:")
    print(categories)

# Preprocessing of data
nclasses = len(categories)
AY2 = keras.utils.to_categorical(AY, nclasses)
QY2 = keras.utils.to_categorical(QY, nclasses)

if DEBUG:
    index = 50
    # Show an element of data set
    plt.imshow(AX[index], cmap="gray", interpolation="nearest")
    plt.show()
    print("> AY[{}]: ", index, AY[index])
    print("> Category of image: ", categories[AY[index]])
    print("> AY2[{}]: ", index, AY2[index])
    print("> AY2: ", AY2.shape, AY2.dtype)

# Normalizing data
nl, nc = AX.shape[1], AX.shape[2]  # 28, 28
AX = AX.astype('float32')/255.0  # 0 a 1
QX = QX.astype('float32')/255.0  # 0 a 1


# -----------Modelo da rede neuronal---------------------------

model = Sequential()
model.add(Flatten(input_shape=(nl, nc)))
model.add(Dense(400, activation='sigmoid'))
model.add( Dense(100, activation='relu') )
model.add( Dense(nclasses, activation='softmax') )

# Plot model
plot_model(model, to_file='mnist-draw-model.png', show_shapes=True)

model.summary()  # opt=optimizers.sgd(lr=0.5)
opt = optimizers.Adam()

model.compile(optimizer=opt,
              loss='mse',
              # loss='categorical_crossentropy',
              metrics=['accuracy'])

H = model.fit(AX, AY2,
              batch_size=10,
              epochs=NUM_EPOCHS,
              verbose=True)


QP = model.predict(QX)
QP = QP.argmax(axis=-1)

score = model.evaluate(QX, QY2, verbose=False)
print('> Test loss: ', score[0])
print('> Test accuracy: ', score[1])

# Save model
model.save('mmnist-fashion-model.h5')

# f = plt.figure()

fig, ax = plt.subplots(4, 5)

plt.suptitle('Fashion-MNIST classificão', fontsize=20)

for i in range(4):
    for j in range(5):
        ax[i, j].text(0, -4, categories[QY[4*i+j + 123]], color="b", fontsize=12)
        ax[i, j].text(0, -1, categories[QP[4*i+j + 123]], color="r", fontsize=12)

        if (categories[QY[4*i+j + 123]] != categories[QP[4*i+j + 123]]):
            color_img = cv2.cvtColor(QX[4*i+j + 123], cv2.COLOR_GRAY2RGB)
            color_img[np.where((color_img == [1, 1, 1]).all(axis=2))] = [1.0, 0.5, 0.4]
            ax[i, j].imshow(color_img)
            ax[i, j].axis("off")
        else:
            ax[i, j].imshow(QX[4*i+j + 123], cmap="gray")
            ax[i, j].axis("off")

plt.savefig("mnist-fashion-classificator.png")
plt.show()
