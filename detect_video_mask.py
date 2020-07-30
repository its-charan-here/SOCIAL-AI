import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import draw_outputs


flags.DEFINE_string('classes', './data/labels/coco.names', 'path to classes file')
flags.DEFINE_string('weights', './weights_mask/yolov3.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('video', 'test.mp4',
                    'path to video file or number for webcam)')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_integer('num_classes', 3, 'number of classes in the model')



def main(_argv):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    if FLAGS.tiny:
        yolo = YoloV3Tiny(classes=FLAGS.num_classes)
    else:
        yolo = YoloV3(classes=FLAGS.num_classes)

    yolo.load_weights(FLAGS.weights)
    logging.info('weights loaded')

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')

    times = []

    try:
        vid = cv2.VideoCapture(int(FLAGS.video))
    except:
        vid = cv2.VideoCapture(FLAGS.video)

    out = None

    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))
    fps = 0.0
    count = 0
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))  
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fr=0
    while True:
        _, img = vid.read()

        if img is None:
            logging.warning("Empty Frame")
            time.sleep(0.1)
            count+=1
            if count < 3:
                continue
            else: 
                break

        img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        img_in = tf.expand_dims(img_in, 0)
        img_in = transform_images(img_in, FLAGS.size)

        t1 = time.time()
        boxes, scores, classes, nums = yolo.predict(img_in)
        fps  = ( fps + (1./(time.time()-t1)) ) / 2
        tot=0
        for i in range(nums[0]):
            tot+=1

        boxes = np.array(boxes)[0]
        mask=0
        no_mask=0
        for i in range(tot):
            # print(boxes[i])

            x1,y1,x2,y2 = boxes[i]
            y1 = int(y1*height)
            y2 = int(y2*height)
            x1 = int(x1*width)
            x2 = int(x2*width)
            
            # print(x1,x2,y1,y2)
            if class_names[int(classes[0][i])]=="Mask":
                mask+=1
                img = cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
            else:
                no_mask+=1
                img = cv2.rectangle(img, (x1, y1), (x2, y2), (88, 43, 237), 2)
                    
        fr+=1
        print(" Frame number :",fr)


        overlay = img.copy()
        output = img.copy()

        img=cv2.rectangle(overlay, (0, 0), (0+(len("Not Wearing Mask : 100"))*17, 110), (0,0,0), -1)

        img = cv2.putText(img, "Total People : {:.0f}".format(tot), (0, 30),
                        cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1)
    
        img = cv2.putText(img, "Wearing Mask : {:.2f}".format(mask), (0 , 60),
                        cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255), 1)
        
        img = cv2.putText(img, "Not Wearing Mask : {:.0f}".format(no_mask), (0 , 90),
                        cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255), 1)
        
        alpha = 0.5

        cv2.addWeighted(overlay, alpha, output, 1 - alpha,
        0, output)
        
        if FLAGS.output:
            out.write(output)
        cv2.imshow('output', output)   
        if cv2.waitKey(1) == ord('q'):
            break
    vid.release()
    if FLAGS.output:
        out.release()
        # list_file.close()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
