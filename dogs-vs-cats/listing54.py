import os
import shutil

# unzip dogs-vs-cats.zip -d dogs-vs-cats
# unzip train.zip
original_dataset_dir =  '/home/fabricio/projects/usp-projects/image-processing-deep-learnig/dogs-vs-cats'

base_dir = '/home/fabricio/projects/usp-projects/image-processing-deep-learnig/cats_and_dogs_small'
# os.mkdir(base_dir)

train_dir =  os.path.join(base_dir, 'train')
# os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
# os.mkdir(validation_dir)
test_dir = os.path.join(base_dir, 'test')
# os.mkdir(test_dir)

train_cats_dir = os.path.join(train_dir, 'cats')
# os.mkdir(train_cats_dir)

train_dogs_dir = os.path.join(train_dir, 'dogs')
# os.mkdir(train_dogs_dir)

validation_cats_dir = os.path.join(validation_dir, 'cats')
# os.mkdir(validation_cats_dir)

validation_dogs_dir = os.path.join(validation_dir, 'dogs')
# os.mkdir(validation_dogs_dir)

test_cats_dir = os.path.join(test_dir, 'cats')
# os.mkdir(test_cats_dir)

test_dogs_dir = os.path.join(test_dir, 'dogs')
# os.mkdir(test_dogs_dir)

# Fix new hierarchy of Kaggle folder
original_dataset_dir_train = os.path.join(original_dataset_dir, 'train')

# Cats
fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir_train, fname)
    dst = os.path.join(train_cats_dir, fname) 
    shutil.copyfile(src, dst)

fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir_train, fname)
    dts = os.path.join(validation_cats_dir, fname)
    shutil.copyfile(src, dts)

fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir_train, fname)
    dts = os.path.join(test_cats_dir, fname)
    shutil.copyfile(src, dts)

# Dogs
fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir_train, fname)
    dst = os.path.join(train_dogs_dir, fname) 
    shutil.copyfile(src, dst)

fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir_train, fname)
    dts = os.path.join(validation_dogs_dir, fname)
    shutil.copyfile(src, dts)

fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir_train, fname)
    dts = os.path.join(test_dogs_dir, fname)
    shutil.copyfile(src, dts)
