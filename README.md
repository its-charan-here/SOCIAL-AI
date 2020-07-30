# COVID101

[![License](https://img.shields.io/badge/License-Apache2-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

Our vision is to effectively use Artificial Intelligence to provide automated surveillance and ensure protective measures.


## Contents

1. [Short description](#short-description)
1. [Demo video](#demo-video)
1. [The architecture](#the-architecture)
1. [Long description](#long-description)
1. [Project roadmap](#project-roadmap)
1. [Getting started](#getting-started)
1. [Running the tests](#running-the-tests)
1. [Live demo](#live-demo)
1. [Built with](#built-with)
1. [Authors](#authors)
1. [Acknowledgments](#acknowledgments)

## Short description

### What's the problem?

COVID-19 spreads from person to person even before the symptoms start. The best way to prevent the spread is maintaining a social distance of 6 feet among the individuals. But knowingly or unknowingly the people tend to defy the guidelines imposed by our government. Besides Manual supervision of people by authorities in public areas is both difficult and hazardous moreover there isn’t any practical solution in the market for Small and medium-sized enterprises, which leads to poor decision making during the pandemic.


### The idea

In order to solve these critical problems, we present u COVID101, A technological solution to tackle health precautionary measurements. An automated real-time social distancing and Face Mask detection among the people will not only make the job of the authorities more safe and easy but also remind the public to follow the required guidelines. This can enable effortless regulation among the people facilitating the small scale companies to recommence without compromising safety.

With the Real-time graphical analysis of the situation in a given area like the average distance among the people, no of people not following social distancing, Social distancing Index. Will help us to understand the necessary steps that need to be taken in order to flatten the curve.

This software is customizable to hardware components too. Social distancing detection is combined with drones for seamless surveillance in restricted areas.

## Demo video

[![](https://raw.githubusercontent.com/its-charan-here/COVID101/master/temp/thumb.PNG?token=AKXUHA5SPL66LI36BJUHR3C7FPVEM)](https://youtu.be/8iOfzc6Wrqo)

## The architecture

![Video transcription/translation app](https://github.com/its-charan-here/COVID101/blob/master/temp/arch.jpg)

1. The live footage from CCTV surveillance systems and Drones is used as an input.
2. The footages will be processed in real-time on a cloud server having a GPU.
3. With the real time video processing, the data generated is stored in a separate csv file.

## Long description

[More detail is available here](https://docs.google.com/presentation/d/1LgQmfI-UdoYnCt_wXGfXAWPUgXm8-CXeJIdRPtDQv2Q/edit?usp=sharing)

## Project roadmap

![example](https://github.com/its-charan-here/COVID101/blob/master/temp/roadmap.png)

## Getting started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites and Installing

Install `anaconda with python3` from https://www.anaconda.com/products/enterprise/

```bash
# Tensorflow CPU
conda env create -f conda-cpu-env.yml
conda activate cpu-env

# Tensorflow GPU
conda env create -f conda-gpu-env.yml
conda activate gpu-env
```
#### Downloading official pretrained weights of YOLOv3 - Social Distancing Detection
You can download the yolov3 weights by clicking [here](https://pjreddie.com/media/files/yolov3.weights) then save them to the weights folder.

#### Saving your yolov3 weights as a TensorFlow model.
Load the weights using `load_weights.py` script. This will convert the yolov3 weights into TensorFlow .ckpt model files!
```
# yolov3
python load_weights.py
```
#### Downloading official pretrained weights of YOLOv3 - Face Mask Detection 

You can download the yolov3 - Face Mask Detection weights by clicking [here](https://drive.google.com/drive/folders/1crqeQoxHkjwFJ89esmWcCtVDmEGmN-nn?usp=sharing) then save them to the weights_mask folder.

#### Saving your yolov3 weights as a TensorFlow model.
Load the weights using `load_weights_mask.py` script. This will convert the yolov3 weights into TensorFlow .ckpt model files!
```
# yolov3
python load_weights_mask.py
```

## Running the tests

Explain how to run the automated tests for this system


### Test with a video - Social Distancing

Exectute the file `detect_video.py` script. This will execute the same social distancing algorithms and give a live output on the screen . use the flag `--output` for saving the output file with your prefered name.
uncomment line number `277` to view the live graph outputs. 
```bash
python detect_video.py --video test.mp4 --output test_result.mp4 --output_csv test_csv.csv
```
### Test with a video - Face Mask Detection

Identify the presence of Face Mask among the people in a video by executing `detect_video_mask.py` script. This will yield a live output on your screen.
```bash
python detect_video_mask.py --video test_mask.mp4 --output test_mask_output.mp4
```

## Built with

* [Python](https://python.org) - Programming language which has been used
* [Yolo v3](https://pjreddie.com/darknet/yolo/) - To handle object detection

## Authors

* **Poojan Panchal** - [PoojanPanchal](https://github.com/PoojanPanchal)

## License

This project is licensed under the Apache 2 License - see the [LICENSE](LICENSE) file for details

## Acknowledgments

* Based on [The AI Guys](https://github.com/theAIGuysCode/Object-Detection-API).
