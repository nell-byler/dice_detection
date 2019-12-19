# Real-time dice detection and classification [WIP]
**Project Goal**: Detect and classify six-sided dice from images, mobile devices, and video.

I play a lot of Warhammer 40k, a dice-based tabletop board game, and enjoy watching live-streamed tournament games on Twitch. A decent streaming setup for 40k usually includes two top-down cameras: one for viewing the entire table, and one aimed at a dice tray. Many aspects of the game are determined by dice rolls, and each player will roll many dice at a time in the shared dice tray. The use of a dice camera significantly improves the viewing experience (*"Exciting! Player 1 rolled a 12-inch charge!"*), but screen real estate is expensive and the dice camera is often relegated to a small section of the overall view, which can make it difficult to see the results of any given roll.

After seeing some of the super cool [results](https://medium.com/tensorflow/training-and-serving-a-realtime-mobile-object-detector-in-30-minutes-with-cloud-tpus-b78971cf1193) for real-time object detection and classification, a dice detector and classifier for streamed games seemed like a natural extension of the application. Eventually, I would like a system that can also post-processes a few statistics (e.g., the total sum of all visible dice faces, the number of 1s, number of 2s) to output to the screen in a more visible manner...*but first, a working dice detection model!*

![phone img](/img/Screenshot_20191219_094842.jpg)

![phone gif](/img/giphy.gif)

## Table of Contents
1. [Proof-of-Concept: image classification with CNN](README.md#proof-of-concept)
2. [Object detection using own dataset](README.md#object-detection)
   - [Generating a labeled dataset and model](README.md##dataset-and-model)
   - [Running on a local machine with GPU](README.md##local-machine)
   - [Running on GCS AI Platform with TPU](README.md##GCS)
   - [Running on AWS EC2 GPU with docker container](README.md##AWS)
3. [Extension to mobile devices with TFLite](README.md#TFlite)
4. [Extension to video applications](README.md#video-footage)

# Proof of Concept
There was one existing dataset on kaggle with dice images. I wanted to train a Convolutional Neural Network (CNN) to classify images of single six-sided dice as either 1, 2, 3, 4, 5, or 6, and I was curious how difficult it would be to implement and train classifier from scratch. Turns out, not too long! You can look at the results in [0_dice-classification.ipynb](1_data_relabel_train_test_split.ipynb). The CNN architecture was heavily inspired by LeNet-5, and took about 2 hours to run on my macbook.

# Object Detection
Image classification is a relatively straight forward concept, but object detection and localization is a much more complex problem. This part of the project was significantly more involved than the initial proof-of-concept, and required that I generate and label my own dataset. I followed EdjeElectronic's [very comprehensive tutorial](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10#8-use-your-newly-trained-object-detection-classifier), with a few modifications. I briefly outline the steps below.

## Dataset and Model
### Process overview
This process closely follows the official [tensorflow guide](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md) for object detection, described below. *Note - I have included rough time estimates for each step, so that those who are following along get an idea of which parts are the most time-consuming. These are based on a first time walk-through, so if you have done this before, they are likely an overestimate!*

1. I took 250 pictures of rolled dice (`data/JPEGImages/`). [*30min*]
2. I used [labelImg](https://github.com/tzutalin/labelImg) to generate labels for each image (`data/Annotations/`). [*3hrs*]
3. I generated test and train sets, and translated YOLO label format to one digestible by TF (see [jupyter notebook](1_data_relabel_train_test_split.ipynb)). [*1hr*]
4. I generated a label file for TF (see `data/dice_label_map.pbtxt`). [*5min*]
5. I generated TFRecord files for the train and test sets (see [jupyter notebook](2_generate_TFRecord_hack.ipynb)). [*1hr*]
6. I downloaded a pre-trained model from the TensorFlow detection [model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md). [*5min*] *Note - I later went back and spent much more time researching the best model for my particular application, see below*
   - The mobile device applications shown here are only available for SSD class models.
   - To take advantage of GCS TPU resources, you will need a TPU compatible model - for more information see [this article](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tpu_compatibility.md)).
   - I initially chose to use the `ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03` model, because I wanted higher accuracy classifications and larger input images. *However* - I later realized that this model cannot be reformatted into the quantized output format required for TFLite. **If you want to do the mobile device application with GCS, save yourself the headache and use either of the two quantized, TPU-compliant SSD models (which you can find in the list of the full model zoo [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)).**
   - I put the downloaded model in `models/ssd_mobilenet_v1_quantized_300x300_coco14_sync_2018_07_18`.
7. I configured the input file for transfer learning, following the official tensorflow [guide](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/configuring_jobs.md). [*1hr*]
   - See specific instructions below.

### Step 7 in detail: the configuration file
This requires some significant modifications to the config file (see `models/ssd_mobilenet_fpn.config`), as described below.
This is for the `ssd_mobilenet_v1_quantized_300x300_coco14_sync_2018_07_18` model, and I am modifying the config file from `models/research/object_detection/samples/config/ssd_mobilenet_v1_quantized_300x300_coco14_sync.config`.

At L12, specify the number of classes:
```
num_classes=6
```
In the `image_resizer` section at L48 (optional, but 300px was too small for accurate results with my particular dataset):
```
height: 640
width: 640
```
In the `train_config` section at L141:
```
fine_tune_checkpoint:<path_to_model>/model.ckpt
batch_size=32
```
*Note - I tried `batch_size=64` and ran out of memory on GCS, and on my local machine I used `batch_size=8`.*

In the `train_input_reader` section at L173:
```
label_map_path: "<path_to_data>/dice_label_map.pbtxt"
input_path: "<path_to_tfrecords>/train.record"
```
In in the `eval_input_reader` section at L186:
```
label_map_path: "<path_to_data>/dice_label_map.pbtxt"
input_path: "<path_to_tfrecords>/test.record"
```
In the `eval_config` section at L180, update to reflect the size of your test/eval set:
```
num_examples: 50
```

**AWESOME!** Now you're ready to run your pre-trained model on your new dataset. This can be done on your [local machine](README.md##local-machine), or on the cloud - either [AWS](README.md##AWS) or [GCS](README.md##GCS)...*choose your own adventure, dear reader!*

## Local Machine
### Implementation using docker containers
Using docker containers is a nifty way to make use of the GPU capabilities of tensorflow *without* having to dig too deeply into the dependencies between CUDA, cuDNN, and various versions of tensorflow - I initially attempted to do this project using tensorflow 2.0, and eventually had to revert to 1.14 for all of the object detection applications; using docker containers made this switch *nearly* trivial.

*Specifics:* These instructions are for a linux machine running ubuntu 16.04 with the latest docker container library (19.03) and NVIDIA container runtime. I followed the official tensorflow [guide](https://www.tensorflow.org/install/docker) for installation with docker containers. *Note - I have a fairly old NVIDIA GPU (GTX960), and found that model training took much longer than I wanted, which ultimately led me to try out cloud compute resources (see sections on AWS and GCS below).*

### Initial container setup
Download the appropriate docker image for your desired tensorflow/python combination. I wanted to use an image with tensorflow v1.14 with python3 and jupyter access, but the full list of available images is available [here](https://hub.docker.com/r/tensorflow/tensorflow/tags/)):
```
docker pull tensorflow/tensorflow:1.14.0-gpu-py3-jupyter
```

I wanted to access data on my local machine from within the container, so I needed to mount a volume for the docker container:
```
sudo docker volume create nell-dice
```
This creates a folder at `/var/lib/docker/volumes/nell-dice`, which I soft linked to a more easily accessible directory (`~/d6`). The data and files stored here are permanent (i.e., they won't disappear when you stop the container) and are accessible by both the container and your local machine. *Note - unless you add your user to the docker group, any interaction with the containers will require sudo access! If you encounter any issues with read/write privileges for the volume, your user must be an owner of the entire docker directory.*

Then, to start up a container, use the following command:
```
sudo docker run -it --runtime=nvidia -p 8888:8888 --mount source=nell-dice,target=/d6 tensorflow/tensorflow:1.14.0-gpu-py3-jupyter bash
```

### Creating a container image prepared for object detection
I wanted to create a new image of the container that was fully set-up for object detection - this makes deployment to AWS machines in the [later section](README.md##AWS) much simpler. To do so, I followed the official tensorflow object detetion [installation instructions](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md).

After launching the container, update apt-get and install the following packages:
```
apt update
apt install emacs
pip install --user Cython
pip install --user contextlib2
pip install --user pillow
pip install --user lxml
```

Grab the relevant repositories (tensorflow models, cocoapi):
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

Test your tensorflow installation (*Note - this command will only work with tensorflow 1.XX, NOT 2.XX!*):
```
python object_detection/builders/model_builder_test.py
```

Finally, create new docker image:
```
sudo docker container list
sudo docker commit <id_of_your_container> <image_nickname>
```

**Hooray!** Now you can run your new docker container anywhere:
```
sudo docker run -it --runtime=nvidia -p 8888:8888 --mount source=nell-dice,target=/d6 <image_nickname> bash
cd /d6/models/research
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
python object_detection/builders/model_builder_test.py
```

**Other helpful commands:**
- To run jupyter notebook session from within docker container:
```
jupyter notebook --ip 0.0.0.0 --no-browser --allow-root
```
- To open a new terminal in the same container:
```
sudo docker container list
sudo docker exec -it <id_of_container> /bin/bash
```

### Testing out object detection installation
See if you can run through the [object_detection_tutorial.ipynb](https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb). Ignore the first few cells of the notebook, which try to install tensorflow/pycocotools/do a protobuf compilation -- you have just done this. *Note - as mentioned by EdjeElectronics, I had to comment out L29, L30 in `object_detection/utils/visualization_utils.py` to get the images to display inline in the notebook.*

### Training the SSD_mobilenet model
Assuming your directory structure follows the above instructions, you can train your model like so:
```
# from models/research/

python object_detection/model_main.py \
--pipeline_config_path=/d6/dice_detection/models/ssd_mobilenet_v1_quantized_300x300_coco14_sync_2018_07_18/pipeline.config \
--model_dir=/d6/dice_detection/models/ssd_mobilenet_v1_quantized_300x300_coco14_sync_2018_07_18/train \
--alsologtostderr \
--num_train_steps=200000
```
*Note - to get this particular SSD model to run on my local machine with its aging GPU, I had to decrease the batch_size from 64 to 8, which in turn required an increase in the number of training steps. After running for 10 hours, the model still hadn't reached acceptable loss levels (0.5; should be more like 0.03).*

### Visualizing the results of trained model
Check out the results of the trained model in [3_visualize_trained_model_output.ipynb](3_visualize_trained_model_output.ipynb)!

![example model results](/img/out_IMG_20191209_095447.jpg)

*Not bad at all! After looking through several images, I did find a few mysterious mis-labels, but mostly confined to difficult cases - oblique viewing angles, or dice with difficult/low-contrast colors (e.g., red pips on transluscent blue/black plastic). It'd be worthwhile to go back and add some more training images with oblique viewing angles. Eventually.*

## GCS
To run the model on Google Cloud Services AI platform, I closely followed [this](https://medium.com/tensorflow/training-and-serving-a-realtime-mobile-object-detector-in-30-minutes-with-cloud-tpus-b78971cf1193) TensorFlow tutorial. The first half of the tutorial (Google Cloud setup, installing tensorflow, installing object detection, uploading dataset to GCS) was very straight forward. I was able to follow the tutorial word-for-word on both my linux machine and my macbook pro, so I won't reproduce that text here. *Note - if this is your first time using GCS, make sure you take advantage of the current offer for a $300 credit for new users!*

I did modify the command used to kick off training, to reflect that `ml-engine` is now `ai-platform` and that I'd like to use python3 (L1 and L5, respectively):
```
gcloud ai-platform jobs submit training dice_object_detection_`date +%s` \
--job-dir=gs://dice_detection/train \
--packages dist/object_detection-0.1.tar.gz,slim/dist/slim-0.1.tar.gz,/tmp/pycocotools/pycocotools-2.0.tar.gz \
--module-name object_detection.model_tpu_main \
--python-version 3.5 \
--runtime-version 1.14 \
--scale-tier BASIC_TPU \
--region us-central1 \
-- \
--model_dir=gs://dice_detection/train \
--tpu_zone us-central1 \
--pipeline_config_path=gs://dice_detection/data/pipeline.config
```

### A few model benchmarks
- The `ssd_mobilenet_v1_fpn` model (`batch_size=64`, image size of 640x640, and 25K training steps) took about 3 hours to run and reached a final loss of 0.04. It cost $11.
- The `ssd_mobilenet_v1_quantized` model (`batch_size=32`, image size of 640x640, 50k training steps) took about 3 hours to run and reached a final loss of 0.1. It cost $16.

## AWS
----- **_SECTION IN PROGRESS!_** -----

Create a base instance in AWS with a small instance type for a clean ubuntu 18.04 operating system. You'll want to start with a small instance so you can build complexity on an ultra-cheap machine, and then port it over to a larger instance.

### Upload docker image to ECR repository
We will use the docker image we [previously set up](README.md###creating-a-container-image-prepared-for-object-detection).

```
pip install awscli --upgrade
sudo $(aws ecr get-login --no-include-email --region us-west-2)
sudo docker image list
sudo docker tag <something> <something>.dkr.ecr.us-west-2.amazonaws.com/ml-dice:latest
sudo docker push <something>.dkr.ecr.us-west-2.amazonaws.com/ml-dice:latest
```

# TFLite
### Deploying your trained model to a mobile device!
This portion of the project follows the tensorflow object detection tutorial, found [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_mobile_tensorflowlite.md). I will note that the linked tutorial is fairly brief, and this was the most fiddly part of the project by far, so I will go through my steps in a bit more detail.

1. Download Android Studio and get example program working [*30min*]
2. Building TF from source [*1hr + 3hrs for compilation*]
3. Export trained model for mobile device [*1hr*]

## Download Android Studio and test example program
This part is pretty straight forward. Clone the tensorflow "examples" repository to your machine (`$TF_EXAMPLES`):
```
git clone https://github.com/tensorflow/examples.git
```
Make sure your android device is set up with developer tools and USB debugging (if you're not sure, check [here](https://developer.android.com/studio/debug/dev-options)). At some point before actually running the project on Android Studio, connect your phone to your computer.

Download and install Android Studio ([here](https://developer.android.com/studio/index.html)). This should also install the SDK tools, build-tools, and platfoorm-tools, which you can check under the menu's `tools > SDK manager` after launching.

Within Android Studio, open the example project (`$TF_EXAMPLES/lite/examples/object_detection/android`). When I did this, Android Studio automatically started building the project, which required downloading a plug-in or two. After the project has successfully built, make sure your phone is recognized as a device (you should see its name at the top of the project), and run the project. If your phone screen is unlocked, a "TensorFlowLite" app will pop up and you can start detecting/classifying objects around you. Pretty cool.

## Building TensorFlow from source
To output your trained model to a format digestible by a mobile device, you'll need to install tensorflow from source, which I did following TensorFlow's "install from source" [guide](https://www.tensorflow.org/install/source). I will note that this part of the project was very time-consuming, and I ultimately went for a bare bones installation (ignoring the existance of my GPU card entirely) to simplify the process.

Per the instructions, you may need to install a few python packages via pip. I did not and went directly to the next step - downloading tensorflow from source.

Download tensorflow from source, and checkout your desired release (v1.14):
```
git clone https://github.com/tensorflow/tensorflow.git
cd tensorflow
git checkout r1.14
```

For tensorflow v1.14, you will need bazel with a version between `0.24.1` and `0.25.2` (I had to google around a bit to figure this out). In general, tensorflow uses a very old version of bazel. This means you won't be able to install bazel with apt-get and will have to manually install your specific release from their [repository](https://github.com/bazelbuild/bazel/releases). I went with bazel version `0.25.1`, which I downloaded and installed in `~/bin` (already in my `$PATH`).

To install bazel:
```
cd ~/bin
chmod +x bazel-<version>-installer-linux-x86_64.sh
./bazel-0.25.0-installer-linux-x86_64.sh --user
```

Next you'll need to configure the tensorflow package for the bazel build. From `path/to/tensorflow/`:
```
./configure
```
I used the default response (mostly "N") for each step - there are many tutorials detailing this part of the process, for different OS and GPU configurations if you get stuck! Double check that the script has found the version of python and the corrent python library path.

Build the pip package. If `gcc --version` is greater than 5, you'll need to include the additional flag at the end (this command took at least 3 hours):
```
bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0"
```

Build the package:
```
./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
```

and finally, install the package:
```
pip install /tmp/tensorflow_pkg/tensorflow-version-tags.whl
```

## Exporting model for mobile device
At this point, we will return to following along the original tensorflow object detection [guide](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_mobile_tensorflowlite.md).

Assuming you have a trained model with quantized outputs, set the following environment variables:
```
export CONFIG_FILE=/path/to/model/train/pipeline.config
export CHECKPOINT_PATH=/path/to/model/train/model.ckpt-50000
export OUTPUT_DIR=/tmp/tflite
```

The, from `models/research/` run:
```
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

python object_detection/export_tflite_ssd_graph.py \
--pipeline_config_path=$CONFIG_FILE \
--trained_checkpoint_prefix=$CHECKPOINT_PATH \
--output_directory=$OUTPUT_DIR \
--add_postprocessing_op=true
```
Check that this has created two files in `/tmp/tflite`. 

From `path/to/tensorflow/`:
```
bazel run --config=opt tensorflow/lite/toco:toco -- \
--input_file=$OUTPUT_DIR/tflite_graph.pb \
--output_file=$OUTPUT_DIR/detect.tflite \
--input_shapes=1,640,640,3 \
--input_arrays=normalized_input_image_tensor \
--output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3' \
--inference_type=QUANTIZED_UINT8 \
--mean_values=128 \
--std_values=128 \
--change_concat_input_ranges=false \
--allow_custom_ops
```
Your `/tmp/tflite` directory should now contain three files: `detect.tflite`,  `tflite_graph.pb`, and `tflite_graph.pbtxt`.

## Creating and running your new mobile model
1. Copy your `detect.tflite` file to the `assets` directory in the example android project:
```
cp /tmp/tflite/detect.tflite \
  $TF_EXAMPLES/lite/examples/object_detection/android/app/src/main/assets/
```
2. Create a new `labelmap.txt` file in the same folder (`$TF_EXAMPLES/lite/examples/object_detection/android/app/src/main/assets/labelmap.txt`). The contents of the file should be:
```
???
one
two
three
four
five
six
```
3. Make sure the builder will use your files. Comment out the line in `android/app/build.gradle` (near L40) that tells the builder to use the downloaded files:
```
//apply from:'download_model.gradle'
```
4. If you used a non-standard image size, you will need to modify the size of input images in `DetectorActivity.java`, near L53. I did this in Android Studio, but for reference, the file is located in `android/app/src/main/java/org/tensorflow/lite/examples/detection/DetectorActivity.java`.
```
  private static final int TF_OD_API_INPUT_SIZE = 640;
```
5. Clean and rebuild the project in Android Studio: Menu > Build > Clean Project; and Menu > Build > Rebuild Project.
6. Run on your device! *Weeeee!*

![android](/img/Screenshot_20191219_095010.jpg)

# Video footage

---- **_WRITE UP IN PROGRESS!_ ** ----
