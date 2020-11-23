# Udacity CVND Image Captioning Project
* Udacity Computer Vision Nanodegree Image Captioning Project

## Introduction
* The repository contains a neural network, which can automatically generate captions from images. 
* Instructions how to setup<br>
First we need pycocotools to be setuped to load coco dataset images and corresponding annotations<br>
1. _Create a new environment_<br>
```
conda create -n <envName><br>
```
2. _Activate the environment_<br>
```
conda activate <envName><br>
```
3. _Install cython_<br>
```
pip install cython<br>
```
4. _Install git_<br>
```
conda install -c anaconda git<br>
```
5. _Install pycocotools from this GitHub rep_<br>
```
pip install git+https://github.com/philferriere/cocoapi.git#egg=pycocotools^&subdirectory=PythonAPI
```
  
* Download some specific data from here: http://cocodataset.org/#download (described below)

1. Under **Annotations**, download:
   **2014 Train/Val annotations [241MB]** (extract captions_train2014.json and captions_val2014.json, and place at locations cocoapi/annotations/captions_train2014.json and cocoapi/annotations/captions_val2014.json, respectively)  
   **2014 Testing Image info [1MB]** (extract image_info_test2014.json and place at location cocoapi/annotations/image_info_test2014.json)

2. Under **Images**, download:
   **2014 Train images [83K/13GB]** (extract the train2014 folder and place at location cocoapi/images/train2014/)
   **2014 Val images [41K/6GB]** (extract the val2014 folder and place at location cocoapi/images/val2014/)
   **2014 Test images [41K/6GB]** (extract the test2014 folder and place at location cocoapi/images/test2014/)
   
* Place cocoapi into opt folder created in project parent path. For more detailed steps follow the instruction described here [here](https://github.com/cocodataset/cocoapi) just don't remember to place cocoapi under opt!




## Network Architecture
The solution architecture consists of:
1. CNN encoder, which encodes the images into the embedded feature vectors:
![image](https://github.com/Antanskas/ud-Image_captioning/blob/master/repo_images/encoder.PNG)
2. Decoder, which is a sequential neural network consisting of LSTM units, which translates the feature vector into a sequence of tokens:
![image](https://github.com/Antanskas/ud-Image_captioning/blob/master/repo_images/decoder.PNG)

## Results
These are some of the outputs give by the network using the [COCO dataset](http://cocodataset.org/):

* Validation dataset prediction with calculated BLEU calculated averaging over 1000 validation samples

![example1](https://github.com/Antanskas/ud-Image_captioning/blob/master/repo_images/validation_example.PNG)

* Test dataset prediction

![example2](https://github.com/Antanskas/ud-Image_captioning/blob/master/repo_images/test_example.PNG)

* Snipped of training log

![example3](https://github.com/Antanskas/ud-Image_captioning/blob/master/repo_images/training_log.PNG)

