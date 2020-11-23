# Udacity CVND Image Captioning Project
Udacity Computer Vision Nanodegree Image Captioning Project

## Introduction
The repository contains a neural network, which can automatically generate captions from images. 

## Network Architecture
The solution architecture consists of:
1. CNN encoder, which encodes the images into the embedded feature vectors:
![image](https://github.com/Lexie88rus/Udacity-CVND-Image-Captioning/raw/master/assets/encoder.png)
2. Decoder, which is a sequential neural network consisting of LSTM units, which translates the feature vector into a sequence of tokens:
![image](https://github.com/Lexie88rus/Udacity-CVND-Image-Captioning/raw/master/assets/decoder.png)

## Results
These are some of the outputs give by the network using the [COCO dataset](http://cocodataset.org/):

Validation dataset prediction with calculated BLEU calculated averaging over 1000 validation samples

![example1](https://github.com/Antanskas/ud-Image_captioning/blob/master/repo_images/validation_example.PNG)

Test dataset prediction

![example2](https://github.com/Antanskas/ud-Image_captioning/blob/master/repo_images/test_example.PNG)

Snipped of training log
![example3](https://github.com/Antanskas/ud-Image_captioning/blob/master/repo_images/training_log.PNG)

