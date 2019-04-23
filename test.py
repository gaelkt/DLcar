# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 18:38:14 2019

@author: Gael Kamdem De Teyou
"""
########################### This file is used to test the model
import gc
gc.collect()
import tensorflow as tf
import numpy as np
import scipy.io as sio
import cv2
import math
import os
import time
import scipy.io
import sys
sys.path.insert(0, 'lib/')
from GoogleNetwork import GoogLeNet as DNN


# Set this path to your dataset directory
directory = 'datasets/KingsCollege/'
dataset = 'dataset_test.txt'

###############                Parameters
number_testing_images = 343
number_channels = 3

############       Results to save
error_position = np.zeros((number_testing_images))  #position error iterated on samples
theta = np.zeros((number_testing_images))   #orientation error iterated on samples
duration = np.zeros((number_testing_images)) #computation time iterated on samples

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



def reading_testing_data():
    y_test = np.zeros((number_testing_images, 7))
    X_test = np.zeros((number_testing_images, 224, 224, number_channels))
    i = 0

    with open(directory+dataset) as f:
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
            y_test[i,:] = [p0,p1,p2,p3,p4,p5,p6]
            X_test_now = cv2.imread(directory+image_name)
            X_test_now = resizing_images(X_test_now, 224, cropping=0)
            #X_train.append(X_train_now)
            X_test[i, :, :, :] = X_test_now
            i = i+1
    return X_test, y_test


#############            Normalization
    #We normalize the data by substracting the mean and scaling 
def normalization_test(X_test, y_test, MEAN):
    # # Forcing the pixels and the poses to be coded as floats
    X_test = X_test.astype('float32')
    y_test = y_test.astype('float32')
    #substracting the mean
    X_test2 = X_test - X_test.mean(axis=(0,1,2),keepdims=1)
    #Scaling
    X_test2 /= 1    # X_test2 /= 255 
    return X_test2, y_test


   
      
##############################################################################
##############################################################################
##############################################################################

# Reset the graph
tf.reset_default_graph()

#Placeholder input data
image = tf.placeholder(tf.float32, [1, 224, 224, number_channels], name="image_data")

#Architecture
net = DNN({'data': image})

#Output of DNN
# Last softmax transformed to FC
position_pred = net.layers['cls3_fc_pose_xyz']
orientation_pred= net.layers['cls3_fc_pose_wpqr']


# Initializer variable
init = tf.global_variables_initializer()

saver = tf.train.Saver()

#Getting testing images and parameters
print('Reading the data')
X_test, y_test= reading_testing_data()

#Getting mean of training images
path_save = "Save"
filename = 'MEAN.mat'
x = sio.loadmat(os.path.join(path_save, filename))
MEAN = x['MEAN']

#Normalizing
X_test, y_test = normalization_test(X_test, y_test, MEAN)
print("Shape of y_test ", np.shape(y_test))
print("Shape of Dataset X_test ", np.shape(X_test))


gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8833)
print('Starting training')
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(init)
    
    # Loading the model
    saver = tf.train.import_meta_graph('Save/model.ckpt.meta')
    saver.restore(sess, "Save/model.ckpt")
    
    for i in range(number_testing_images):
        # Timer starts
        start = time.time()
        # Input image
        image_test = X_test[i, :, :, :]
        image_test_tensor = np.reshape(image_test, [1, 224, 224, number_channels])
        feed = {image: image_test_tensor}
        
        #Estimated parameters
        position_pred_result, orientation_pred_result = sess.run([position_pred, orientation_pred], feed_dict=feed)
        
        # Timer ends
        end = time. time()
        duration[i] = end-start 
        
        #True parameters
        position_true = y_test[i, 0:3]
        orientation_true = y_test[i, 3:7]

        #Errors on each sample
        # Error on position
        error_position[i] = np.linalg.norm(position_true-position_pred_result)  
        #Error on oriantation
        q1 = orientation_true / np.linalg.norm(orientation_true)
        q2 = orientation_pred_result / np.linalg.norm(orientation_pred_result)
        d = abs(np.sum(np.multiply(q1,q2)))
        theta[i] = 2 * np.arccos(d) * 180/math.pi
        print('Sample:  ', i, '  Error position in m:  ', error_position[i], '  Error orientation in deg: ', theta[i])

    
    median_error_position = np.median(error_position)
    median_error_orientation = np.median(theta)
    median_duration = np.median(duration)
    
    #Saving the data
    scipy.io.savemat('Save/Test/error_position.mat', mdict={'error_position': error_position})
    scipy.io.savemat('Save/Test/theta.mat', mdict={'theta': theta})
    scipy.io.savemat('Save/Test/duration.mat', mdict={'duration': duration})
    
    print('Median error on position in meters ', median_error_position)
    print('Median error on orientation in degrees ', median_error_orientation)
    print('Median computation latence in ms ', 1000*median_duration)
    

    

print('Finish')   