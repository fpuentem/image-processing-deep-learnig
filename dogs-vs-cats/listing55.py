# Listing 5.5 Instantiating a small convenet for dogs vs. cats classification  
from keras import layers, metrics
from keras import models

model = models.Sequential()
model.add( layers.Conv2D( 32, (3, 3), activation='relu',
                          input_shape=(150, 150, 3) ) )
model.add( layers.MaxPool2D( (2,2) ))
model.add( layers.Conv2D( 64, (3,3), activation='relu') )
model.add( layers.MaxPooling2D( (2,2) ))
model.add( layers.Conv2D( 128, (3,3), activation='relu') )
model.add( layers.MaxPooling2D( (2,2) ))
model.add( layers.Conv2D( 128, (3,3), activation='relu') )
model.add( layers.MaxPooling2D( (2,2) ))
model.add( layers.Flatten() )
model.add( layers.Dense( 512, activation='relu' ) )
model.add( layers.Dense( 1, activation='sigmoid' ) )

model.summary()

# Listing 5.6 Configuring the model for training
from keras import optimizers

model.compile( loss='binary_crossentropy',
               optimizer=optimizers.RMSprop(lr=1e-4),
               metrics=['accuracy'])

# Listing 5.7 Using ImageDataGenerator to read images from directories

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_dir = '/home/fabricio/projects/usp-projects/image-processing-deep-learnig/cats_and_dogs_small/train'
test_dir = '/home/fabricio/projects/usp-projects/image-processing-deep-learnig/cats_and_dogs_small/test'
validation_dir = '/home/fabricio/projects/usp-projects/image-processing-deep-learnig/cats_and_dogs_small/validation'

train_generator = train_datagen.flow_from_directory(    train_dir,
                                                        target_size=(150, 150),
                                                        batch_size=20,
                                                        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(validation_dir,
                                                        target_size=(150, 150),
                                                        batch_size=20,
                                                        class_mode='binary')
        
# Listing 5.8: Fitting the model using the bach generator
history = model.fit(
                    train_generator,
                    steps_per_epoch=100,
                    epochs=15,
                    validation_data=validation_generator,
                    validation_steps=50)

# Listing 5.9: Savinf the model
model.save('cats_and_dogs_small_1.h5')

# Listing 5.10: Displaying curves of losses and accuracy during training
import matplotlib.pyplot as plt
# ['accuracy', 'loss', 'val_accuracy', 'val_loss']
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()