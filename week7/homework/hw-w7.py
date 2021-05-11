# flann.py
# # Veja https://stackoverflow.com/questions/8301962/opencv-pythons-api-flannbasedmatcher
import tensorflow.keras as keras
from keras.datasets import mnist
import cv2
import numpy as np
import sys
import time# the data, split between train and test sets

(AX, ay),(QX, qy) = mnist.load_data()
print(AX.shape)
print(ay.shape)
print(QX.shape)
print(qy.shape)

N = 5

# Increase the data training set

AX_inc = np.empty((N*AX.shape[0], 28, 28))

# original
for i in range( AX.shape[0] ):
    AX_inc[i] = AX[i]

# north
for i in range( AX.shape[0], 2*AX.shape[0] ):  
    tmp = AX[i - AX.shape[0]]
    a = np.concatenate((tmp[1:,:], np.array([tmp[0,:]])), axis=0)
    AX_inc[i] = a

# south
for i in range( 2*AX.shape[0], 3*AX.shape[0] ):  
    tmp = AX[i - 2*AX.shape[0]]
    # print(tmp[-1,:])
    # print(np.array([tmp[:-1,:]]))
    a = np.concatenate((np.array([tmp[-1,:]]), tmp[:-1,:]), axis=0)
    AX_inc[i] = a

# east
for i in range( 3*AX.shape[0], 4*AX.shape[0] ):  
    tmp = AX[i - 3*AX.shape[0]]
    # print(tmp[:,1:])
    # print(np.array([tmp[:,0]]))
    a = np.concatenate((tmp[:,1:], np.array([tmp[:,0]]).reshape((28,1))), axis=1)
    AX_inc[i] = a

# west
for i in range( 3*AX.shape[0], 4*AX.shape[0] ):  
    tmp = AX[i - 3*AX.shape[0]]
    # print(tmp[:,1:])
    # print(np.array([tmp[:,0]]))
    a = np.concatenate((np.array([tmp[:,-1]]).reshape((28,1)), tmp[:,:-1] ), axis=1)
    # print(a.shape)
    AX_inc[i] = a


# Training data
ax = np.empty((N*AX.shape[0], 14, 14))

for i in range( N*AX.shape[0] ):  
    ax[i] = cv2.resize( AX[i % AX.shape[0]], (14, 14) )


# Test data
qx = np.empty( (QX.shape[0], 14, 14) )

for i in range( QX.shape[0] ):  
    qx[i] = cv2.resize( QX[i], (14, 14) )

ax = ax.astype('float32')
qx = qx.astype('float32')
ax = ax.reshape(ax.shape[0],ax.shape[1]*ax.shape[2])
qx = qx.reshape(qx.shape[0],qx.shape[1]*qx.shape[2])
ax /= 255.0 #0 a 1
qx /= 255.0 #0 a 1
qp = np.empty(qy.shape, dtype="uint8")

t1 = time.time()
FLANN_INDEX_KDTREE = 1;# bug: flann enums are missing
flann_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 64)
flann = cv2.flann_Index(ax, flann_params)

t2 = time.time()
matches, dists = flann.knnSearch(qx, 256)
t3 = time.time()

print(matches.shape)
for l in range( matches.shape[0] ):  
    i = matches[l,0]
    print(i) 
    qp[l] = ay[i % AX.shape[0]] # here

erros = 0
    
for l in range( matches.shape[0] ):
    if( qp[l] != qy[l] ): 
        erros += 1

print("Erros=%5.2f%%"%(100.0*erros/qy.shape[0]))
print("Tempo de treinamento: %f"%(t2-t1))
print("Tempo de predicao: %f"%(t3-t2))

# # Listing 2.6 Displaying the fourth digit
# import matplotlib.pyplot as plt

# # digit = AX[20, :, :]
# # digit = digit.reshape((28,28))

# digit = ax[3*AX.shape[0] - 20, :, :]
# digit = digit.reshape((14,14))


# plt.imshow(digit, cmap = plt.cm.binary)
# plt.show()