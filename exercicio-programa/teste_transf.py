# teste_zero.py
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH']='true'

from keras.models import load_model
from keras.preprocessing import image
from keras import optimizers
import numpy as np

from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt

# prepare iterators
test_dir = '/home/fabricio/projects/usp-projects/image-processing-deep-learnig/cats_and_dogs_small/test'

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(test_dir,
                                                    target_size=(150, 150),
                                                    batch_size=1,
                                                    class_mode='binary')

# load the model we saved
model = load_model('./transf.h5')
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['accuracy'])

# evaluate model
_, acc = model.evaluate(test_generator, steps=len(test_generator), verbose=0)
print()
print('> Accuracy of model transf.h5(%):')
print('>> %.3f' % (acc * 100.0))
print()

# {'cats': 0, 'dogs': 1}
classes = test_generator.class_indices

key_list = list(classes.keys())
val_list = list(classes.values())

# Plot first 10 wrong predictions
i = 1
for _ in range(1000):
    x, y = test_generator.next()
    pred = model.predict_classes(x)

    plt.figure(i)
    if(y[0].astype("int32") != pred[0][0]):
        imgplot = plt.imshow(image.array_to_img(x[0]))
        
        print("Predicted: {}".format(key_list[val_list.index(pred)]))
        print("Label: {}".format(key_list[val_list.index(y[0].astype("int32"))]))
        
        plt.text(60, 10, key_list[val_list.index(pred)], color='red', fontsize='xx-large') # bbox=dict(fill=False, edgecolor='red', linewidth=4))
        i += 1
        plt.show()
    
    if( i%11 == 0 ):
        break
