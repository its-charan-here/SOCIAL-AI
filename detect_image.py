import time
import math
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images, load_tfrecord_dataset
from yolov3_tf2.utils import draw_outputs
from PIL import Image, ImageDraw, ImageFont



flags.DEFINE_string('classes', './data/labels/coco.names', 'path to classes file')
flags.DEFINE_string('weights', './weights/yolov3.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_list('images', 'img.png', 'list with paths to input images')
flags.DEFINE_string('tfrecord', None, 'tfrecord instead of image')
flags.DEFINE_string('output', './detections/', 'path to output folder')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')

def mid_point(img,person,idx,height,width):
    #get the coordinates
    x1,y1,x2,y2 = person[idx]
    y1 = int(y1*height)
    y2 = int(y2*height)
    x1 = int(x1*width)
    x2 = int(x2*width)
    _ = cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)

    #compute bottom center of bbox
    x_mid = int((x1+x2)/2)
    y_mid = int(y2)
    mid   = (x_mid,y_mid)

    # _ = cv2.circle(img, mid, 5, (0, 0, 255), -1)
    # cv2.putText(img, str(idx), mid, cv2.FONT_HERSHEY_SIMPLEX,1, (255, 255, 255), 2, cv2.LINE_AA)

    return mid

from scipy.spatial import distance

def compute_distance(midpoints,num):
    dist = np.zeros((num,num))
    for i in range(num):
        for j in range(num):
            dst = distance.euclidean(midpoints[i], midpoints[j])
            dist[i][j]=dst
    return dist

def find_closest(dist,num,thresh):
    p1=[]
    p2=[]
    d=[]
    for i in range(num):
        for j in range(i,num):
            if( (i!=j) & (dist[i][j]<=thresh) & (dist[j][i]<=thresh)):
                print(dist[i][j], dist[j][i])
                p1.append(i)
                p2.append(j)
                d.append(dist[i][j])
    return p1,p2,d

def change_2_red(img,person,p1,p2,height,width):
    risky = np.unique(p1+p2)
    
    for i in risky:
        x1,y1,x2,y2 = person[i]
        y1 = int(y1*height)
        y2 = int(y2*height)
        x1 = int(x1*width)
        x2 = int(x2*width)
        # class_name = "Suspect!"
        color = (88, 43, 237)
        
        # cv2.putText(img, 'Suspect!', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        _ = cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)  
        # _=cv2.rectangle(img, (x1, y1-30), (x1+(len(class_name))*17, y1), color, -1)
        # _=cv2.putText(img, class_name,(x1, y1-10),0, 0.75, (255,255,255),2)
            
    return img, len(risky)

def height_dist(img,person,idx,height,width):
    #get the coordinates
    x1,y1,x2,y2 = person[idx]
    y1 = int(y1*height)
    y2 = int(y2*height)

    #compute height of the person
    pheight = y2-y1
    return pheight



def main(_argv):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    if FLAGS.tiny:
        yolo = YoloV3Tiny(classes=FLAGS.num_classes)
    else:
        yolo = YoloV3(classes=FLAGS.num_classes)

    yolo.load_weights(FLAGS.weights).expect_partial()
    print('weights loaded')

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    print('classes loaded')

    if FLAGS.tfrecord:
        dataset = load_tfrecord_dataset(
            FLAGS.tfrecord, FLAGS.classes, FLAGS.size)
        dataset = dataset.shuffle(512)
        img_raw, _label = next(iter(dataset.take(1)))
    else:
        raw_images = []
        images = FLAGS.images
        for image in images:
            img_raw = tf.image.decode_image(
                open(image, 'rb').read(), channels=3)
            height = img_raw.shape[0]
            width = img_raw.shape[1]
            raw_images.append(img_raw)
    num = 0    
    print("raw image :", raw_images)
    for raw_img in raw_images:
        num+=1
        img_in = tf.expand_dims(raw_img, 0)
        img_in = transform_images(img_in, FLAGS.size)

        t1 = time.time()
        boxes, scores, classes, nums = yolo(img_in)
        t2 = time.time()
        logging.info('time: {}'.format(t2 - t1))

        print('detections:')
        tot=0
        for i in range(nums[0]):
            if(class_names[int(classes[0][i])]=='person'):
                tot+=1
                print('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                                np.array(scores[0][i]),
                                                np.array(boxes[0][i])))

    
        
        #identity only persons 
        ind = np.where(classes[0]==0)[0]
        # print(ind)

        #identify bounding box of only persons
        boxes1 =np.array(boxes) 
        person=boxes1[0][ind]

        #total no. of persons
        num= len(person)

        
        img = cv2.imread('img.png')
        
        midpoints = [mid_point(img,person,i,height,width) for i in range(tot)] 
        
        heights_of_people = [height_dist(img,person,i,height,width) for i in range(tot)]

        print("\n\nHeights :", heights_of_people)
        print("Avg height : ",)

        if(len(heights_of_people)!=0):
            avg = sum(heights_of_people)/len(heights_of_people)

        print("\n\nMidpoints:",midpoints)
        dist= compute_distance(midpoints,tot)

        print("\n\ndistance : ", dist)
        
        if avg >= 100 :
            avg = avg * 0.85
        thresh=avg

        p1,p2,d=find_closest(dist,tot,thresh)

        for i in range(len(p1)):
            cv2.line(img, midpoints[p1[i]], midpoints[p2[i]], (88, 43, 237),2)
        

        img,count = change_2_red(img,person,p1,p2,height,width)
    
        df = pd.DataFrame({"p1":p1,"p2":p2,"dist":d})
        print(df)


        total_interaction = int((tot*(tot-1))/2)
        faulty_interaction = len(p1)
        sd_index =(faulty_interaction/total_interaction)*100 

        print(sd_index)

        overlay = img.copy()
        output = img.copy()

        img=cv2.rectangle(overlay, (0, 0), (0+(len("Not following : 100"))*17, 80), (0,0,0), -1)

        img = cv2.putText(img, "Total People : {:.0f}".format(tot), (0, 30),
                          cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1)
       
        # img = cv2.putText(img, "Following : {:.0f}".format(tot-count), (0 , 60),
        #                   cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255), 1)
        
        img = cv2.putText(img, "Not Following : {:.0f}".format(count), (0 , 60),
                          cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255), 1)
        
        alpha = 0.5

        cv2.addWeighted(overlay, alpha, output, 1 - alpha,
		0, output)
        
        cv2.imshow('output', output)


        key = cv2.waitKey(20000)
        if key == 27:#if ESC is pressed, exit loop
            cv2.destroyAllWindows()

            
        # cv2.imwrite(FLAGS.output + 'detection_avg_changing' + '.jpg', img)

        

        print("height : ", height)
        print("width : " ,  width)
        
        


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass