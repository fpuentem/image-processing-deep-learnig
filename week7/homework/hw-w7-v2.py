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

# Configuration parameters
N = 5 # 1
resize_dig = 14
TREES = 64 # 4
NEIGHBOURS = 256 # 1

ax = np.empty((N*AX.shape[0], resize_dig, resize_dig))

print()
print("*** Size of dataset : {} ***".format(N*AX.shape[0]))
# original
for i in range( AX.shape[0] ):
    # AX_inc[i] = AX[i]
    ax[i] = cv2.resize( AX[i % AX.shape[0]], (resize_dig, resize_dig) )

# Increase the size training set
print("*> Increasing the size of data set ...")
if N > 1:
    # north
    for i in range( AX.shape[0], 2*AX.shape[0] ):  
        tmp = cv2.resize( AX[i - AX.shape[0]], (resize_dig,resize_dig) )
        a = np.concatenate((tmp[1:,:], np.array([tmp[0,:]])), axis=0)
        ax[i] = a

    # south
    for i in range( 2*AX.shape[0], 3*AX.shape[0] ):  
        tmp = cv2.resize( AX[i - 2*AX.shape[0]], (resize_dig,resize_dig) )
        # print(tmp[-1,:])
        # print(np.array([tmp[:-1,:]]))
        a = np.concatenate((np.array([tmp[-1,:]]), tmp[:-1,:]), axis=0)
        ax[i] = a

    # east
    for i in range( 3*AX.shape[0], 4*AX.shape[0] ):  
        tmp = cv2.resize( AX[i - 3*AX.shape[0]], (resize_dig,resize_dig) )
        # print(tmp[:,1:])
        # print(np.array([tmp[:,0]]))
        a = np.concatenate((tmp[:,1:], np.array([tmp[:,0]]).reshape((resize_dig,1))), axis=1)
        ax[i] = a

    # west
    for i in range( 4*AX.shape[0], 5*AX.shape[0] ):  
        tmp = cv2.resize( AX[i - 4*AX.shape[0]], (resize_dig, resize_dig) )
        # print(tmp[:,1:])
        # print(np.array([tmp[:,0]]))
        a = np.concatenate((np.array([tmp[:,-1]]).reshape((resize_dig,1)), tmp[:,:-1] ), axis=1)
        
        ax[i] = a


# Test data
qx = np.empty( (QX.shape[0], resize_dig, resize_dig) )

for i in range( QX.shape[0] ):  
    qx[i] = cv2.resize( QX[i], (resize_dig, resize_dig) )

ax = ax.astype('float32')
qx = qx.astype('float32')
ax = ax.reshape(ax.shape[0], ax.shape[1]*ax.shape[2])
qx = qx.reshape(qx.shape[0], qx.shape[1]*qx.shape[2])
ax /= 255.0 #0 a 1
qx /= 255.0 #0 a 1
qp = np.empty(qy.shape, dtype="uint8")


# Train KDtree
print("*> Trainig the KDTREE ...")
t1 = time.time()
FLANN_INDEX_KDTREE = 1;# bug: flann enums are missing
flann_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = TREES) 
flann = cv2.flann_Index(ax, flann_params)


# Search our KDtree with 256 neighbors
t2 = time.time()
matches, dists = flann.knnSearch(qx, NEIGHBOURS)
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
print("*>> Erros=%5.2f%%"%(100.0*erros/qy.shape[0]))
print("*>> Tempo de treinamento: %f sec."%(t2-t1))
print("*>> Tempo de predicao: %f sec."%(t3-t2))


# # Listing 2.6 Displaying the fourth digit
# import matplotlib.pyplot as plt

# # digit = AX[20, :, :]
# # digit = digit.reshape((28,28))

# digit = ax[5*AX.shape[0] - 200, :, :]
# # digit = digit.reshape((resize_dig,resize_dig))


# plt.imshow(digit, cmap = plt.cm.binary)
# plt.show()