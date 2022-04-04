# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from PIL import Image
import time

import yolo_v3
import yolo_v3_tiny

from utils import load_coco_names, draw_boxes, get_boxes_and_inputs, get_boxes_and_inputs_pb, non_max_suppression, \
                  load_graph, letter_box_image

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'input_img', 'test00001.jpg', 'Input image')
tf.app.flags.DEFINE_string(
    'output_img', 'res00002.jpg', 'Output image')
tf.app.flags.DEFINE_string(
    'class_names', './models/yolov3_custom_2022_02_03_01.names', 'File with class names')
tf.app.flags.DEFINE_string(
    'weights_file', './models/yolov3_custom_2022_02_03_01.weights', 'Binary file with detector weights')
tf.app.flags.DEFINE_string(
    'data_format', 'NHWC', 'Data format: NCHW (gpu only) / NHWC')
tf.app.flags.DEFINE_string(
    'ckpt_file', '', 'Checkpoint file')  # ./saved_model/yolov3_custom_2022_02_03_01.ckpt
tf.app.flags.DEFINE_string(
    'frozen_model', './saved_model/frozen_yolov3_custom_2022_02_03_01.pb', 'Frozen tensorflow protobuf model')
tf.app.flags.DEFINE_bool(
    'tiny', False, 'Use tiny version of YOLOv3')
tf.app.flags.DEFINE_bool(
    'spp', False, 'Use SPP version of YOLOv3')

tf.app.flags.DEFINE_integer(
    'size', 416, 'Image size')

tf.app.flags.DEFINE_float(
    'conf_threshold', 0.1, 'Confidence threshold')
tf.app.flags.DEFINE_float(
    'iou_threshold', 0.4, 'IoU threshold')

tf.app.flags.DEFINE_float(
    'gpu_memory_fraction', 1.0, 'Gpu memory fraction to use')

def main(argv=None):

    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_memory_fraction)

    config = tf.compat.v1.ConfigProto(
        gpu_options=gpu_options,
        log_device_placement=False,
    )

    img = Image.open(FLAGS.input_img)
    img_resized = letter_box_image(img, FLAGS.size, FLAGS.size, 128)
    img_resized = img_resized.astype(np.float32)
    classes = load_coco_names(FLAGS.class_names)

    if FLAGS.frozen_model:

        t0 = time.time()
        frozenGraph = load_graph(FLAGS.frozen_model)
        print("Loaded FROZEN graph in {:.2f}s".format(time.time()-t0))

        boxes, inputs = get_boxes_and_inputs_pb(frozenGraph)

        with tf.compat.v1.Session(graph=frozenGraph, config=config) as sess:
            t0 = time.time()
            detected_boxes = sess.run(
                boxes, feed_dict={inputs: [img_resized]})

    else:
        if FLAGS.tiny:
            model = yolo_v3_tiny.yolo_v3_tiny
            print("USE tiny version of Yolo model")
        elif FLAGS.spp:
            model = yolo_v3.yolo_v3_spp
            print("Use SPP version of YOLOv3")
        else:
            model = yolo_v3.yolo_v3
            print("Use SIMPLE version of YOLOv3")

        boxes, inputs = get_boxes_and_inputs(model, len(classes), FLAGS.size, FLAGS.data_format)

        saver = tf.train.Saver(var_list=tf.global_variables(scope='detector'))

        with tf.Session(config=config) as sess:
            t0 = time.time()
            saver.restore(sess, FLAGS.ckpt_file)
            print('Model restored in {:.2f}s'.format(time.time()-t0))

            t0 = time.time()
            detected_boxes = sess.run(
                boxes, feed_dict={inputs: [img_resized]})

    filtered_boxes = non_max_suppression(detected_boxes,
                                         confidence_threshold=FLAGS.conf_threshold,
                                         iou_threshold=FLAGS.iou_threshold)
    print("Predictions found in {:.2f}s".format(time.time() - t0))
    print("Boxes:", filtered_boxes)

    draw_boxes(filtered_boxes, img, classes, (FLAGS.size, FLAGS.size), True)

    img.save(FLAGS.output_img)


if __name__ == '__main__':
    tf.compat.v1.app.run()
