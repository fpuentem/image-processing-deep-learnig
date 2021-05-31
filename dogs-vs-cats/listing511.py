# Listing 5.11 Setting up a data augmentation 
# configuration via ImageDataGenerator
from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Listing 5.12 Displaying some randomly augmented training images
from keras.preprocessing import image
import matplotlib.pyplot as plt
import os
import shutil
train_cats_dir = '/home/fabricio/projects/usp-projects/image-processing-deep-learnig/cats_and_dogs_small/test/cats'
fnames = [  os.path.join(train_cats_dir, fname) for
            fname in os.listdir(train_cats_dir)]
# Chooses one image to augment 
img_path = fnames[10]

# Reads the image and resizes it
img = image.load_img(img_path, target_size=(150, 150))

# Converts it to a Numpy array with shape (150, 150, 3)
x = image.img_to_array(img)

# Reshapes it to (1, 150, 150, 3)
x = x.reshape((1,) + x.shape)

i = 0
for batch in datagen.flow(x, batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(image.array_to_img(batch[0]))
    i += 1
    if( i%8 == 0 ):
        break

plt.show()
