# Posenet
This repository provides a TensorFlow implementation for Real-Time 6-DOF Camera Relocalization

- The folder weight contains the pretrained GoogleNet weights that are using for training
- The folder lib contains the GoogleNet architecture and all the function used in the network
- train.py is the training file. You need to create a "datasets" folder where you will put the traning images. This code provide the implementation for the King College dataset that can be downloaded  at this link https://www.repository.cam.ac.uk/handle/1810/251342

After training the model is saved with the name model.ckpt that you can used for testing.

- test.py is the testing file. 

Requirements:

- Tensorflow 1.13
- Opencv
