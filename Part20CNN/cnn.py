# We will import images using keras
# Basically folders banana cats and dogs ke, it will understand it is a dog or a cat
# This dataset a sort of benchmark for deep learning models
# Full dataset on Kaggle
# feature scalling is a must for deep learning models
# No pre processing needed, done by making folders and stuff

# Part 1 : Build CNN
"""
Seq : to initialise neural network(2 ways to initialise, as graph or layers, we use sequential)
Conv2d : first step of cnn, when we add convolution layers
videos in 3d, time is 3rd dimension
MaxPooling2D: pooling layers
Flatten : convert pool feature maps into large feature vector as input
Dense : used to add fully connected layers in CNN
"""

from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator

# Initialise CNN
classifier = Sequential()

# Step 1 : Convolution
# Fully connected ke liye dense use kiya tha na, isme conv2d
# Convolution kernel = feature detector
# No. of feature detectors bhi daalna hai : usually start with 32, then next mein 64, and 128
# But working on CPU so will start slowly
# No. of rows and columns of feature matrix bhi dalna hai function
# border_mode is how to handle the borders of the images, default is 'same'
# input shape is shape of input images, e.g same size ya same format nahi hai sabka
# input_shape mein, first is no. of channels (coloured mein 3(R,G,B ke liye), black and white mein 1)
# other 2 are dimensions of that array for each channel
# ye kaafi time consuming, so choosing smaller things,
# input_shape=(3,64,64) used in Theanos backend, but we use tensorflow, so order is (64,64,3)
# activation function to prevent -ve pixel values, inorder to have non linearity

# 32 feature maps of 3X3 matrices
classifier.add(Convolution2D(32, 3, 3,
                             border_mode='same',
                             input_shape=(64, 64, 3),
                             activation='relu'))

# Pooling, maximum of 4 cells lenge har baar : Max Pooling
# Applied to reduce no. of nodes in the next layer
# Will make it less computation intensive
# pool_size usually 2X2

classifier.add(MaxPooling2D(pool_size=(2, 2)))

# for better accuracy, add 2nd CNN layer, no input_shape needed
classifier.add(Convolution2D(32, 3, 3,
                             border_mode='same',
                             activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Flattening : make vector of features, make input layer of ann
# High numbers are always maintained in all the steps, and these are the key attributes

classifier.add(Flatten())

# Full connection step
# isme we dont want a small number, because input mein bhot saari hain and koyi rule of thumb nahi hai waise

classifier.add(Dense(output_dim=128, activation='relu'))

# sigmoid for binary, softmax for non binary output
classifier.add(Dense(output_dim=1, activation='sigmoid'))

# Compile CNN
# Loss is logarithmic in nature and is needed in NNs
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fit CNN to images
# image augmentation/image processing : to prevent overfitting of images
# We can either have lots and lots of images
# or apply transformations(randomly) like shifting, flippping and we get many many diverse images
# enrich images with adding a whole lot of images
#use flow_from directory in keras documentation
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        # Dimensions expected by CNN
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        # Path of test Set
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        # no. of images in training set
        steps_per_epoch=8000,
        epochs=25,
        validation_data=test_set,
        validation_steps=2000)
# to achieve better accuracy we have to make NN deeper, how :
# 1 add fully connected layer
# 2 add convolutional layer, much better option
