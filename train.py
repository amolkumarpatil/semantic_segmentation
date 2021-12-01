"""
Train custom binary segmentation model
author: Amolkumar Patil
"""


import tensorflow as tf
import segmentation_models as sm
sm.set_framework('tf.keras')
sm.framework()
import glob
import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

# Default size for the images
size_x = 256
size_y = 256

# Load the images and masks
img_directories = './binary_lane_bdd/Images'
mask_directories = './binary_lane_bdd/Labels'

train_images = []
count = 0

os.chdir(img_directories)
for image_path in glob.glob("*.jpg"):
    imgs = cv2.imread(image_path)
    imgs = cv2.resize(imgs, (size_x, size_y))
    train_images.append(imgs)
    count += 1
    print(count)

count = 0
train_masks = []
os.chdir(mask_directories)
for mask_path in glob.glob("*.jpg"):
    print(mask_path)
    masks = cv2.imread(mask_path, 0)
    masks = cv2.resize(masks, (size_x, size_y))
    train_masks.append(masks)
    count += 1
    print(count)

train_images = np.array(train_images)
train_masks = np.array(train_masks)

X = train_images
Y =train_masks


# Get shape of the data
print(X.shape)
print(Y.shape)


#Split the data
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)

# Define Network
BACKBONE = 'resnet34'
preprocess_input = sm.get_preprocessing(BACKBONE)
x_train = preprocess_input(x_train)
y_train = preprocess_input(y_train)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# define network parameters
BATCH_SIZE = 8
CLASSES = ['lane']
LR = 0.0001
EPOCHS = 100
n_classes = 1
activation = 'sigmoid'

#create model
model = sm.Unet(BACKBONE, classes=n_classes, activation=activation)

# define optimizer
optim = tf.keras.optimizers.Adam(LR)

# Segmentation models losses can be combined together by '+' and scaled by integer or float factor
dice_loss = sm.losses.DiceLoss()
focal_loss = sm.losses.BinaryFocalLoss()
total_loss = dice_loss + (1 * focal_loss)

metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

# Model Compilation and training
# compile keras model with defined optimozer, loss and metrics
model.compile(optim, total_loss, metrics)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint('./best_model.h5', save_weights_only=True, save_best_only=True, mode='min'),
    tf.keras.callbacks.ReduceLROnPlateau(),
]

y_train = tf.cast(y_train, tf.float32)
y_test = tf.cast(y_test, tf.float32)
history = model.fit(x_train, y_train, epochs = EPOCHS,
                    batch_size = BATCH_SIZE, validation_data = (x_test, y_test),
                    callbacks = [callbacks])

model.save('./op.h5')


