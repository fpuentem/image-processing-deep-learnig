#classif1.py
import os; os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow.keras as keras
from keras.preprocessing import image
import numpy as np
import sys
from sys import argv

# Para usar resnet50
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
model = ResNet50(weights='imagenet')
target_size = (224, 224)

# # Para usar InceptionV3
# from keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
# model = InceptionV3(weights='imagenet')
# target_size = (299, 299)

# #Para usar InceptionResNetV2
# from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input, decode_predictions
# model = InceptionResNetV2(weights='imagenet')
# target_size = (299, 299)

img_path = "./tiger.jpg"
#"./rat-topo.jpg"
#"./payaso1.jpg"

img = image.load_img(img_path, target_size=target_size)
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
preds = model.predict(x)
p = decode_predictions(preds, top=3)[0]

# decode the results into a list of tuples (class, description, probability)
#print('Predicted:', p)
# Predicted: [(u'n02504013', u'Indian_elephant', 0.82658225),
#(u'n01871265', u'tusker', 0.1122357), (u'n02504458', u'African_elephant', 0.061040461)]
print("***** Prediction *****")
for predicao in p:
    print("%8.2f%% %s"%(100*predicao[2],predicao[1]))