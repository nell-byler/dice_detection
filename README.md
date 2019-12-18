# dice_detection [WIP]
**AIM**: Detect and classify six-sided dice from images, mobile devices, and video.

I play a lot of Warhammer 40k, a dice-based tabletop board game, and enjoy watching live-streamed tournament games on Twitch. A decent streaming setup for 40k usually includes two top-down cameras: one for viewing the entire table, and one aimed at a dice tray. Many aspects of the game are determined by dice rolls, and each player will roll many dice at a time in the shared dice tray. The use of a dice camera significantly improves the viewing experience ("Exciting! Player 1 rolled a 12-inch charge!"), but screen real estate is expensive and the dice camera is often relegated to a small section of the overall view - making it difficult to see the results of any given roll.

After seeing some of the [super cool results](https://medium.com/tensorflow/training-and-serving-a-realtime-mobile-object-detector-in-30-minutes-with-cloud-tpus-b78971cf1193) for real-time object detection and classification, I thought a dice detector and classifier for streamed games was a natural extension. Eventually, I want a dice detector and classifier that also post-processes a few statistics (the total sum of all visible dice faces, the number of 1s, number of 2s...etc.) which could be output to the screen in a more visible manner...but that will come later.

## Table of Contents
1. [Proof-of-Concept: image classification with CNN](README.md#proof-of-concept)
2. [Object detection using own dataset](README.md#object-detection)
   - [local machine configuration]
   - [AWS EC2 GPU with docker container]
   - [GCS TPU]
3. [TFLite and mobile devices](README.md#TFlite)
4. [Video Applications](README.md#video-footage)

# Proof of Concept
There was one existing dataset on kaggle with dice images. I wanted to train a Convolutional Neural Network (CNN) to classify images of single six-sided dice as either 1, 2, 3, 4, 5, or 6, and I was curious how difficult it would be to implement and train a CNN from scratch. Turns out, not too long! You can look at the results in [0_dice-classification.ipynb](1_data_relabel_train_test_split.ipynb). The CNN architecture was heavily inspired by LeNet-5, and took about 2 hours to run on my macbook.

# Object Detection
This part of the project was significantly more involved, and required that I generate and label my own dataset. I followed EdjeElectronic's [very comprehensive tutorial](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10#8-use-your-newly-trained-object-detection-classifier), with a few modifications. I briefly outline the steps below.



## Implementation on local machine with docker containers
Using docker containers is a nifty way to make use of the GPU capabilities of tensorflow without having to dig too deeply into the dependencies between CUDA, cuDNN, and various versions of tensorflow - I initially attempted to do this project in 2.0, and eventually had to revert to 1.14 for all of the object detection results, using docker containers made this switch nearly trivial.

These instructions are for a linux machine running ubuntu 16.04 with the latest docker container library (19.03) and nvidia container runtime. I followed the tensorflow [guide](https://www.tensorflow.org/install/docker) on using docker containers. Note - I have a fairly old NVIDIA GPU (GTX960), and found that model training took much longer than I wanted, which ultimately led me to try out cloud compute resources (see sections on AWS and GCS below).

### Container Setup
I wanted to access data on my local machine, so I created a volume for the docker container:
```
sudo docker volume create nell-dice
```
This creates a folder in `/var/docker`, which I softlinked to a more accessible directory (`/d6`). The data and files stored here are permanent (won't disappear when you end the container session) and accessible by both the container and your local machine.
*Note - unless you add your user to the docker group, any interaction with the containers will require sudo access! If you encounter any issues with read/write privleges for the volume, your user must be an owner of the entire docker directory.*

```
sudo docker run -it --runtime=nvidia -p 8888:8888 --mount source=nell-dice,target=/d6 tensorflow/tensorflow:1.14.0-gpu-py3-jupyter bash
```

### Creating an object detection ready container image
This portion primarily follows the official object detetion [installation instructions](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md).

From the container, update apt-get, and install the following packages:
```
apt update
apt install emacs
pip install --user Cython
pip install --user contextlib2
pip install --user pillow
pip install --user lxml
```

Get relevant repositories (models, cocoapi):
```
git clone https://github.com/tensorflow/models.git
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
make
cp -r pycocotools <path_to_tensorflow>/models/research/
```

Protobuf compilation:
```
# From tensorflow/models/research/
protoc object_detection/protos/*.proto --python_out=.
```


Update your python path. You will need to do this every time you log in.
```
# from models/research/
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```

Test tensorflow installation:
```
python object_detection/builders/model_builder_test.py
```
*note - this will only work with TF 1.XX, NOT 2.XX!*

Create new docker image:
```
sudo docker container list
sudo docker commit <id_of_your_container> <image_nickname>
```

Now you can run your new docker container like so:
```
sudo docker run -it --runtime=nvidia -p 8888:8888 --mount source=nell-dice,target=/d6 <image_nickname> bash
cd /d6/models/research
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
python object_detection/builders/model_builder_test.py
```

To run jupyter notebook session from within docker container:
```
jupyter notebook --ip 0.0.0.0 --no-browser --allow-root
```

To open a new terminal in the same container session:
```
sudo docker container list
sudo docker exec -it <id_of_container> /bin/bash
```

### Testing out object detection
See if you can run through the [object_detection_tutorial.ipynb](https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb). Ignore the first few cells, where the notebook tries to install tensorflow/pycocotools/protobuf compilation, you have just done this. *Note -- as mentioned by EdjeElectronics, I had to comment out L29, L30 in `object_detection/utils/visualization_utils.py` to get the images to show up in the notebook.*

## AWS

## GCS


# TFLite

# Video footage
