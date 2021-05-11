# flann.py
# # Veja https://stackoverflow.com/questions/8301962/opencv-pythons-api-flannbasedmatcher
import tensorflow.keras as keras
from keras.datasets import mnist
import cv2
import numpy as np
import sys
import time# the data, split between train and test sets

# Load MNIST data
(AX, ay),(QX, qy) = mnist.load_data()


# Increase the size training set

N = 5
ax = np.empty((N*AX.shape[0], 14, 14))

# original
for i in range( AX.shape[0] ):
    # AX_inc[i] = AX[i]
    ax[i] = cv2.resize( AX[i % AX.shape[0]], (14, 14) )

# north
for i in range( AX.shape[0], 2*AX.shape[0] ):  
    tmp = cv2.resize( AX[i - AX.shape[0]], (14,14) )
    a = np.concatenate((tmp[1:,:], np.array([tmp[0,:]])), axis=0)
    ax[i] = a

# south
for i in range( 2*AX.shape[0], 3*AX.shape[0] ):  
    tmp = cv2.resize( AX[i - 2*AX.shape[0]], (14,14) )
    # print(tmp[-1,:])
    # print(np.array([tmp[:-1,:]]))
    a = np.concatenate((np.array([tmp[-1,:]]), tmp[:-1,:]), axis=0)
    ax[i] = a

# east
for i in range( 3*AX.shape[0], 4*AX.shape[0] ):  
    tmp = cv2.resize( AX[i - 3*AX.shape[0]], (14,14) )
    # print(tmp[:,1:])
    # print(np.array([tmp[:,0]]))
    a = np.concatenate((tmp[:,1:], np.array([tmp[:,0]]).reshape((14,1))), axis=1)
    ax[i] = a

# west
for i in range( 3*AX.shape[0], 4*AX.shape[0] ):  
    tmp = cv2.resize( AX[i - 3*AX.shape[0]], (14, 14) )
    # print(tmp[:,1:])
    # print(np.array([tmp[:,0]]))
    a = np.concatenate((np.array([tmp[:,-1]]).reshape((14,1)), tmp[:,:-1] ), axis=1)
    ax[i] = a





# Test data
qx = np.empty( (QX.shape[0], 14, 14) )

for i in range( QX.shape[0] ):  
    qx[i] = cv2.resize( QX[i], (14, 14) )

ax = ax.astype('float32')
qx = qx.astype('float32')
ax = ax.reshape(ax.shape[0], ax.shape[1]*ax.shape[2])
qx = qx.reshape(qx.shape[0], qx.shape[1]*qx.shape[2])
ax /= 255.0 #0 a 1
qx /= 255.0 #0 a 1
qp = np.empty(qy.shape, dtype="uint8")

# Train KDtree
t1 = time.time()
FLANN_INDEX_KDTREE = 1;# bug: flann enums are missing
flann_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 64) 
flann = cv2.flann_Index(ax, flann_params)

# Search our KDtree with 256 neighbors
t2 = time.time()
matches, dists = flann.knnSearch(qx, 256)
t3 = time.time()


for l in range( matches.shape[0] ):  
    i = matches[l,0]
    qp[l] = ay[i % AX.shape[0]] # here

# Calculate error
erros = 0
    
for l in range( matches.shape[0] ):
    if( qp[l] != qy[l] ): 
        erros += 1

# Display error rate and times
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