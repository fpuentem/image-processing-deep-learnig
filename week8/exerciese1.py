# versao2.py
# Imprime versao de Tensorflow, Keras e Keras dentro do Tensorflow
# Tambem imprime se GPU esta funcionando, versao de SO, CPU e RAM
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Devido a bug, pode ser necessario colocar o comando abaixo, 
# quando da erro "Failed to get convolution algorithm"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf
import keras
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0],True)
import sys
import cv2
print("Versao python3:",sys.version)
print("Versao de tensorflow:",tf.__version__)
print("Versao de Keras independente:",keras.__version__)
print("Versao de Keras dentro de tensorflow:",tf.keras.__version__)
print("Versao cv2:",cv2.__version__)
os.system("nvcc --version | grep release")
print()

gpu = tf.test.gpu_device_name()

if gpu == "":
    print("Computador sem GPU.")
else:
    print("Computador com GPU:",tf.test.gpu_device_name())
    from tensorflow.python.client import device_lib  
    devices = device_lib.list_local_devices()
    print("Dispositivos:",[x.physical_device_desc for x in devices if x.physical_device_desc!=""]) 
    # os.system('cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2')  
    os.system('cat /usr/include/cudnn.h | grep "define CUDNN_MAJOR" -A 2')
    # os.system('nvidia-smi')
print()

os.system('lsb_release -a | grep "Description"') # imprime qual é o sistema operacional
os.system('cat /proc/cpuinfo | grep -E "model name"') # especificações de CPU
os.system('cat /proc/meminfo  | grep "Mem"') # especificações de RAM