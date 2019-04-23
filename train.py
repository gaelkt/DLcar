# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 18:38:14 2019

@author: Gael Kamdem De Teyou
"""

import gc
gc.collect()
import tensorflow as tf
import numpy as np
import scipy.io
import cv2
import sys
sys.path.insert(0, 'lib/')
from GoogleNetwork import GoogLeNet as DNN

########################### This file is used to train the data


###############             Filess and Folder locations
directory = '/home/ahsan/PoseNet/datasets/KingsCollege/'   # Specify the location of the dataset
dataset = 'dataset_train.txt'          # File containing information on training images and poses

###############                Parameters
iterations = 30000
batch_size = 61
number_training_images = 1220
number_channels = 3



############       Results to save
loss_position_iteration = np.zeros((iterations))
loss_orientation_iteration = np.zeros((iterations))

#############            Resizing the images
# GoogleNet uses images of sizes 224x224. We have to resize the training data to this size
# For example King college dataset has this shape  width = 455 and height = 256
def resizing_images(initial_image, output_side_length, cropping):
    # initial_image: image that we want to resize
    # output_side_length: the output image is of size output_side_length xoutput_side_length, e.g: 224
    # cropping: parameter that tells the type of cropping to be used:
    #   cropping = 0: No cropping, we simpling resize the image
    #   cropping = 1: we do centered cropping
    #   cropping = 1: rescaling before cropping
    
    width = np.shape(initial_image)[1]
    height = np.shape(initial_image)[0]
    new_height = output_side_length
    new_width = output_side_length
    
    if cropping ==0:
        new_image = cv2.resize(initial_image, (new_height, new_width))
    elif cropping ==1:
        left = (width - new_width)//2
        top = (height - new_height)//2
        right = (width + new_width)//2
        bottom = (height + new_height)//2
        new_image = initial_image[top:bottom, left:right]
    else:
        rescaled_height = 256
        rescaled_width = 256
        if height > width:
            rescaled_height = rescaled_width * height // width
        else:
            rescaled_width = rescaled_height * width // height
        rescaled_image = cv2.resize(initial_image, (rescaled_height,rescaled_width))
        height_offset = (rescaled_height - output_side_length) // 2
        width_offset = (rescaled_width - output_side_length) // 2
        new_image = rescaled_image[height_offset:height_offset + output_side_length, width_offset:width_offset + output_side_length]
    return new_image


#############            Function to read the training data
# The file dataset_train.txt is read and parsed
def reading_training_data():
    # position: 1x3
    # orientation: 1x4
    # y_train = pose = [position orientation]: 1x7
    # X_train: training images
    y_train = np.zeros((number_training_images, 7))
    X_train = np.zeros((number_training_images, 224, 224, number_channels))
    i = 0

    with open(directory+dataset) as f:
        # We skip the three first lines
        next(f)  
        next(f)
        next(f)
        for line in f:
            image_name, p0,p1,p2,p3,p4,p5,p6 = line.split()
            p0 = float(p0)
            p1 = float(p1)
            p2 = float(p2)
            p3 = float(p3)
            p4 = float(p4)
            p5 = float(p5)
            p6 = float(p6)
            y_train[i,:] = [p0,p1,p2,p3,p4,p5,p6]
            X_train_now = cv2.imread(directory+image_name)
            X_train_now = resizing_images(X_train_now, 224, cropping=0)
            X_train[i, :, :, :] = X_train_now
            i = i+1
    return X_train, y_train

#############            Normalization
    #We normalize the data by substracting the mean and scaling 
def normalization(X_train, y_train):
    # Forcing the pixels and the poses to be coded as floats
    X_train = X_train.astype('float32')
    y_train = y_train.astype('float32')
    
    MEAN = np.mean(X_train, axis=(0,1,2))  #Calculating the mean for each channel
    
    X_train2 = X_train - X_train.mean(axis=(0,1,2),keepdims=1)  # Substracting the mean 
    X_train2 /= 1        # Scaling to [-1, 1]  X_train2 /= 255
    return X_train2, y_train, MEAN


#We shuffle the data
def shuffle_data(X_train, y_train):
    
    #Initial order of images
    order = np.arange(number_training_images) 
    
    # New order  when shuffling
    np.random.shuffle(order)
    
    # Shuffle the data
    X_train = X_train[order, :]
    y_train = y_train[order, :]
    
    return X_train, y_train


#We generate the batch
def generate_batch_input_data(X_train, y_train, batch_size):
    
    number_batch = number_training_images // batch_size
    
    while True:
        for i in range(number_batch):
            X_train_batch = X_train[i*batch_size:(i+1)*batch_size, :]
            y_train_batch = y_train[i*batch_size:(i+1)*batch_size, :]
            yield X_train_batch, y_train_batch
        
        
##############################################################################
##############################################################################
##############################################################################

# Reset the graph
tf.reset_default_graph()

#Placeholder input data: image, position and orientation
image_data = tf.placeholder(tf.float32, [batch_size, 224, 224, number_channels], name="image_data")
position_true = tf.placeholder(tf.float32, [batch_size, 3],  name="position_true")
orientation_true = tf.placeholder(tf.float32, [batch_size, 4],  name="orientation_true")

# Deep Neural Network
net = DNN({'data': image_data})

#Output of DNN


# First  FC
position_pred_1 = net.layers['cls1_fc_pose_xyz']
orientation_pred_1= net.layers['cls1_fc_pose_wpqr']

# Second FC
position_pred_2 = net.layers['cls2_fc_pose_xyz']
orientation_pred_2= net.layers['cls2_fc_pose_wpqr']

# Last FC
position_pred_3 = net.layers['cls3_fc_pose_xyz']
orientation_pred_3= net.layers['cls3_fc_pose_wpqr']

#Loss function
beta_param = 150 # Parameter used to balance the position error with the orientation error
# First FC
loss_position_1 = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(position_pred_1, position_true))))
loss_orientation_1 = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(orientation_pred_1, orientation_true))))
loss_1 = loss_position_1 + beta_param*loss_orientation_1

# Second FC
loss_position_2 = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(position_pred_2, position_true))))
loss_orientation_2 = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(orientation_pred_2, orientation_true))))
loss_2 = loss_position_2 + beta_param*loss_orientation_2

#Third FC
loss_position_3 = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(position_pred_3, position_true))))
loss_orientation_3 = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(orientation_pred_3, orientation_true))))
loss_3 = loss_position_3 + beta_param*loss_orientation_3

loss = 0.3*loss_1 + 0.3*loss_2 + loss_3 # weighted sum for googlenet loss function


#Optimizer
opt = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.9, beta2=0.999, epsilon=0.00000001, use_locking=False, name='Adam').minimize(loss)

# Initializer variable
init = tf.global_variables_initializer()

#Getting data
print('Reading the data')
X_train, y_train= reading_training_data()
X_train, y_train, MEAN = normalization(X_train, y_train)

scipy.io.savemat('Save/MEAN.mat', mdict={'MEAN': MEAN})
print("Shape of y_train ", np.shape(y_train))
print("Shape of X_train ", np.shape(X_train))

# Shuffling and setting the batch
X_train, y_train = shuffle_data(X_train, y_train)
new_batch = generate_batch_input_data(X_train, y_train, batch_size)

saver = tf.train.Saver()

outputFile = "/home/ahsan/PoseNet/Save/Posenet.ckpt"

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9133)
print('Starting training')
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(init)
    # Loading the pretrained weights
    net.load('weights/weights.npy', sess)
    for i in range(iterations):
        #Getting the current batch
        print('iteration', i)
        X_train_batch, y_train_batch = next(new_batch)

        feed = {image_data: X_train_batch, position_true: y_train_batch[:,0:3], orientation_true:y_train_batch[:,3:7]}
    
        sess.run(opt, feed_dict=feed)
        
        loss_orientation_iteration[i] = sess.run(loss_orientation_3, feed_dict=feed) # Only the last layer is considered as the prediction
        
        loss_position_iteration[i] = sess.run(loss_position_3, feed_dict=feed)  # Only the last layer is considered as the prediction
        
        if i % 1000 == 0:
            saver.save(sess, outputFile)
            scipy.io.savemat('Save/loss_position_iteration.mat', mdict={'loss_position_iteration': loss_position_iteration})
            scipy.io.savemat('Save/loss_orientation_iteration.mat', mdict={'loss_orientation_iteration': loss_orientation_iteration})
        if i % 100 == 0:
            print('iteration number ', i)
            print(' ----------------------- loss on position ', loss_position_iteration[i])
            print(' ----------------------- loss on orientation ', loss_orientation_iteration[i])
    
    saver.save(sess, outputFile)
    scipy.io.savemat('/home/ahsan/PoseNet/Save/loss_position_iteration.mat', mdict={'loss_position_iteration': loss_position_iteration})
    scipy.io.savemat('/home/ahsan/PoseNet/Save/loss_orientation_iteration.mat', mdict={'loss_orientation_iteration': loss_orientation_iteration})
    

print('end of training')   