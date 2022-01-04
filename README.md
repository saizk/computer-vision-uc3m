# Audio processing, Video processing and Computer vision - UC3M

----------------------------------------------------------------------------------------
## Table of contents
0. [Description and Installation](#Description)
1. [Scale-Space Blob Detector](#1-scale-space-blob-detector)
2. [Melanoma Segmentation](#2-melanoma-segmentation)
3. [Melanoma Classification with CNNs](#3-melanoma-classification-with-cnns)
4. [Object Detection with Faster-RCNN](#4-object-detection-with-faster-rcnn)
5. [Feature Selection for Audio Classification](#5-feature-selection-for-audio-classification)
6. [Audio Speech Recognition with DeepSpeech2](#6-audio-speech-recognition-with-deepspeech2)

## Description
Audio processing, Video processing and Computer Vision Laboratories (**UC3M** - *C2.350.16508*).

## Installation
Create a **Python 3.6** virtual environment and run the following command:
```
pip install -r requirements.txt
```
Or specify the name of the project to install specific requirements.
```
pip install -r <PROJECT NAME>/requirements.txt
```
### Installation PyTorch for CUDA 11.3
**PIP ENVIRONMENT**
```
pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

**CONDA ENVIRONMENT**
```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

## PROJECTS

### 1. Scale-Space Blob Detector
Scale-space blob detector based on the Laplacian of Gaussian (LoG) filter.
Full guideline [here](1-Scale-Space-Blob-Detector/Lab_1_guidelines.pdf).

### 2. Melanoma Segmentation
Pre-processing, segmentation and post-processing for melanoma images using thresholding and clustering techniques.
Full guideline [here](2-Melanoma-Segmentation/Lab_2_guidelines.pdf).

### 3. Melanoma Classification with CNNs
Testing of several CNN architectures for melanoma classification (no melanoma, melanoma, keratosis)
Full lab [here](3-Image-Classification-with-CNNs/Lab_3_solved.ipynb).

### 4. Object Detection with Faster-RCNN
Faster-RCNN implementation for object detection and classification using a subset of the PASCAL VOC 2012 database.
Full lab [here](4-Object-Detection-with-F-RCNN/Lab_4_solved.ipynb).

### 5. Feature Selection for Audio Classification
Feature extraction and selection for classifying dogs and cats audios using SVM.
Full guideline [here](5-Audio-Features-Selection/Lab_5_guidelines.pdf).

### 6. Audio Speech Recognition with DeepSpeech2
Comparison of 3 speech recognition architectures based on DeepSpeech2 altering the GRU layer implementation. 
Full lab [here](6-Deep-Learning-for-ASR/Lab_6_solved.ipynb).
