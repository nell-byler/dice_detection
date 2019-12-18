# Real-time dice detection and classification [WIP]
**Project Goal**: Detect and classify six-sided dice from images, mobile devices, and video.

I play a lot of Warhammer 40k, a dice-based tabletop board game, and enjoy watching live-streamed tournament games on Twitch. A decent streaming setup for 40k usually includes two top-down cameras: one for viewing the entire table, and one aimed at a dice tray. Many aspects of the game are determined by dice rolls, and each player will roll many dice at a time in the shared dice tray. The use of a dice camera significantly improves the viewing experience ("Exciting! Player 1 rolled a 12-inch charge!"), but screen real estate is expensive and the dice camera is often relegated to a small section of the overall view - making it difficult to see the results of any given roll.

After seeing some of the [super cool results](https://medium.com/tensorflow/training-and-serving-a-realtime-mobile-object-detector-in-30-minutes-with-cloud-tpus-b78971cf1193) for real-time object detection and classification, I thought a dice detector and classifier for streamed games was a natural extension. Eventually, I want a dice detector and classifier that also post-processes a few statistics (the total sum of all visible dice faces, the number of 1s, number of 2s...etc.) which could be output to the screen in a more visible manner...but that will come later.

## Table of Contents
1. [Proof-of-Concept: image classification with CNN](README.md#proof-of-concept)
2. [Object detection using own dataset](README.md#object-detection)
   - [local machine configuration](README.md##local-machine)
   - [AWS EC2 GPU with docker container](README.md##AWS)
   - [GCS with TPU](README.md##GCS)
3. [TFLite and mobile devices](README.md#TFlite)
4. [Video Applications](README.md#video-footage)

# Proof of Concept
There was one existing dataset on kaggle with dice images. I wanted to train a Convolutional Neural Network (CNN) to classify images of single six-sided dice as either 1, 2, 3, 4, 5, or 6, and I was curious how difficult it would be to implement and train a CNN from scratch. Turns out, not too long! You can look at the results in [0_dice-classification.ipynb](1_data_relabel_train_test_split.ipynb). The CNN architecture was heavily inspired by LeNet-5, and took about 2 hours to run on my macbook.

# Object Detection
This part of the project was significantly more involved, and required that I generate and label my own dataset. I followed EdjeElectronic's [very comprehensive tutorial](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10#8-use-your-newly-trained-object-detection-classifier), with a few modifications. I briefly outline the steps below.

## Labeling own dataset
See also the official [tensorflow guide](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md).

Following the guide, the process involved:
1. I took 250 pictures of rolled dice (`data/JPEGImages/`).
2. I used labelImg to generate labels for each image (`data/Annotations/`).
3. I generated test and train sets, and translated YOLO label format to one digestible by TF (see [jupyter notebook](1_data_relabel_train_test_split.ipynb)).
4. I generated a label file for TF (see `data/dice_label_map.pbtxt`).
5. I generated TFRecord files for the train and test sets (see [jupyter notebook](2_generate_TFRecord_hack.ipynb)).
6. I downloaded the pre-trained model and configured the input file for additional training, following official [tensorflow guide](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/configuring_jobs.md).
   - I chose to use ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03. For mobile device object detection, you need to use an SSD model, and for taking advantage of GCS TPU resources, you need a TPU compatible model (this model does both - for more information see [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tpu_compatibility.md)).
   - I downloaded the model and put it in `models/ssd_mobilenet_XX`.

Step 6 requires some significant modifications to the config file (see `models/ssd_mobilenet_fpn.config`).
Specifically, the following:
```
num_classes=6
fine_tune_checkpoint:<path_to_model>/model.ckpt
```
In in the `train_input_reader` section:
```
label_map_path: "<path_to_data>/dice_label_map.pbtxt"
input_path: "<path_to_tfrecords>/train.record"
```
In in the `eval_input_reader` section:
```
label_map_path: "<path_to_data>/dice_label_map.pbtxt"
input_path: "<path_to_tfrecords>/test.record"
```
In the `eval_config` section:
```
num_examples: 50
```

**AWESOME!** Now you're ready to run your pre-trained model on your new data! This can be done on your [local machine](README.md##local-machine), or using cloud computing - either [AWS](README.md##AWS) or [GCS](README.md##GCS). Choose your own mystery!

## Local Machine
### Implementation using docker containers
Using docker containers is a nifty way to make use of the GPU capabilities of tensorflow without having to dig too deeply into the dependencies between CUDA, cuDNN, and various versions of tensorflow - I initially attempted to do this project in 2.0, and eventually had to revert to 1.14 for all of the object detection applications; using docker containers made this switch *nearly* trivial.

These instructions are for a linux machine running ubuntu 16.04 with the latest docker container library (19.03) and nvidia container runtime. I followed the [tensorflow guide](https://www.tensorflow.org/install/docker) on using docker containers. *Note - I have a fairly old NVIDIA GPU (GTX960), and found that model training took much longer than I wanted, which ultimately led me to try out cloud compute resources (see sections on AWS and GCS below).*

### Initial container setup
Download the appropriate docker image (full list [here](https://hub.docker.com/r/tensorflow/tensorflow/tags/)):
```
docker pull tensorflow/tensorflow:1.14.0-gpu-py3-jupyter
```

I wanted to access data on my local machine, so I created a volume for the docker container:
```
sudo docker volume create nell-dice
```
This creates a folder at `/var/lib/docker/volumes/nell-dice`, which I softlinked to a more easily accessible directory (`~/d6`). The data and files stored here are permanent (i.e., they won't disappear when you stop the container) and are accessible by both the container and your local machine.
*Note - unless you add your user to the docker group, any interaction with the containers will require sudo access! If you encounter any issues with read/write privleges for the volume, your user must be an owner of the entire docker directory.*

```
sudo docker run -it --runtime=nvidia -p 8888:8888 --mount source=nell-dice,target=/d6 tensorflow/tensorflow:1.14.0-gpu-py3-jupyter bash
```

### Creating an object detection "ready" container image
I wanted to create a new image of the container that was "ready to go" for object detection. This portion primarily follows the official object detetion [installation instructions](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md).

After starting up the container, update apt-get, and install the following packages:
```
apt update
apt install emacs
pip install --user Cython
pip install --user contextlib2
pip install --user pillow
pip install --user lxml
```

Grab relevant repositories (models, cocoapi):
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

Update your python path (you will need to do this every time you log in):
```
# from models/research/
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```

Test your tensorflow installation:
```
python object_detection/builders/model_builder_test.py
```
*Note - this will only work with TF 1.XX, NOT 2.XX!*

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

To open a new terminal in the same container:
```
sudo docker container list
sudo docker exec -it <id_of_container> /bin/bash
```

### Testing out object detection installation
See if you can run through the [object_detection_tutorial.ipynb](https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb). Ignore the first few cells, where the notebook tries to install tensorflow/pycocotools/protobuf compilation, you have just done this. *Note - as mentioned by EdjeElectronics, I had to comment out L29, L30 in `object_detection/utils/visualization_utils.py` to get the images to show up in the notebook.*

### Training the SSD_mobilenet model
Assuming your directory structure is as described above, you can start training your model like so:
```
# from models/research/

python object_detection/model_main.py \
--pipeline_config_path=/d6/dice_detection/models/ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/ssd_mobilenet_fpn.config \
--model_dir=/d6/dice_detection/models/ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/train \
--alsologtostderr \
--num_train_steps=200000
```
*Note - to get this particular SSD model to run on my local machine with its aging GPU, I had to decrease the batch_size from 64 to 16, which then required an increase in the number of training steps. After running for 10 hours, the model still hadn't reached acceptable loss levels (0.5, should be more like 0.03).*


## AWS

## GCS

# TFLite

# Video footage
