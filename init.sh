#!/bin/bash

mkdir -p models
wget -O ./models/coco.names https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
wget -O ./models/yolov3.weights https://pjreddie.com/media/files/yolov3.weights # 416x416
wget -O ./models/yolov3.cfg https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg