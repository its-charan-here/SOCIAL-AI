import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
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
import os 
from os.path import isfile, join


flags.DEFINE_string('classes', './data/labels/coco.names', 'path to classes file')
flags.DEFINE_string('weights', './weights/yolov3.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('video', None,
                    'path to video file or number for webcam)')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')

flags.DEFINE_string('output_csv', 'output.csv', 'The output for csv name')





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

    return mid


def compute_distance(midpoints,num):
    dist = np.zeros((num,num))
    for i in range(num):
        for j in range(i+1,num):
            if i!=j:
                dst = distance.euclidean(midpoints[i], midpoints[j])
                dist[i][j]=dst
    return dist

def find_closest(dist,num,thresh):
    p1=[]
    p2=[]
    d=[]
    for i in range(num):
        for j in range(i,num):
            if( (i!=j) & (dist[i][j]<=thresh)):
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
        color = (88, 43, 237)
        
        _ = cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)  
            
    return img,len(risky)

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

    yolo.load_weights(FLAGS.weights)
    logging.info('weights loaded')

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')

    
    
    print(str(FLAGS.output_csv))
    try:
        vid = cv2.VideoCapture(int(FLAGS.video))
    except:
        vid = cv2.VideoCapture(FLAGS.video)
    
    # by default VideoCapture returns float instead of int
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    
    count = 0
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))  
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    
    fr=0
    avg=1
    avg_d=0
    total_interaction = 0
    faulty_interaction = 0 
    sd_index = 0


    df_tot = {"Frame no" : [],"Total People" : [], "SD Index" : [], "Not Following" : [], "Average Distance (Not Following)" : []}

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)

    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    

    def plt_frame_vs_people(x,y1,y2,y3):
        
        l1, = ax1.plot(x, y1, color='blue',linewidth=2,label ='Total People')
        l2, = ax1.plot(x,y2,color='red',linewidth=2,label='Not Following')
        ax1.set(xlabel='Frames', ylabel='Number of People')
        fig1.suptitle('Real Time\nGraphical Analysis')
        fig1.legend((l1,l2),('Total People','Not Following'))

        fig1.show()
        fig1.canvas.draw()
        
        ax2.plot(x,y3,color='orange',linewidth=2,label='SD Index')
        ax2.set(xlabel='Frames', ylabel='SD Index %')
        fig2.suptitle('Real Time\nSocial Distancing Index')

        fig2.show()
        fig2.canvas.draw()


        ax3.plot(x,y4,color='brown',linewidth=2,label='SD Index')
        ax3.set(xlabel='Frames', ylabel='Distance in Feet')
        fig3.suptitle('Real Time\n Average Distance between Defaulters')

        fig3.show()
        fig3.canvas.draw()      
        
        time.sleep(0.1)
        plt.cla()
    
    

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
        
        #identity only persons 
        ind = np.where(classes[0]==0)[0]
        ind_score = np.where(scores[0]>0.50)[0]

        ind = ind[np.in1d(ind,ind_score)]

        
        tot = len(ind)


        #identify bounding box of only persons
        boxes1 =np.array(boxes) 
        person=boxes1[0][ind]
        
        midpoints = [mid_point(img,person,i,height,width) for i in range(tot)]
        heights_of_people = [height_dist(img,person,i,height,width) for i in range(tot)]
        if(len(heights_of_people)!=0):
            avg = sum(heights_of_people)/len(heights_of_people)

        
        dist= compute_distance(midpoints,tot)

        if avg >= 100 :
            avg = avg * 0.85
        thresh=avg
        
        
        p1,p2,d=find_closest(dist,tot,thresh)

        
        if(len(d)!=0):
            avg_d = sum(d)/len(d)
        avg_dis_feet = 5.75 * avg_d/thresh
        # print("Distance is feet : ", avg_dis_feet)


        for i in range(len(p1)):
            cv2.line(img, midpoints[p1[i]], midpoints[p2[i]], (88, 43, 237),2)

        
        img,count = change_2_red(img,person,p1,p2,height,width)

        total_interaction += int((tot*(tot-1))/2)
        faulty_interaction += len(p1)
        if(total_interaction!=0):
            sd_index =(faulty_interaction/total_interaction)*100 
        
        if(fr%1==0):
            df_tot["Frame no"].append(fr)
            df_tot["Total People"].append(tot)
            df_tot["SD Index"].append(sd_index)
            df_tot["Not Following"].append(count)
            df_tot["Average Distance (Not Following)"].append(avg_dis_feet)

            x = df_tot["Frame no"]
            y1 = df_tot["Total People"]
            y2 = df_tot["Not Following"]
            y3 = df_tot["SD Index"]
            y4 = df_tot["Average Distance (Not Following)"]

            # plt_frame_vs_people(x,y1,y2,y3)


        fr+=1
        print(" Frame number :",fr)
        

        
        
        overlay = img.copy()
        output = img.copy()

        fps  = ( fps + (1./(time.time()-t1)) ) / 2

        print("FPS: ",fps)

        img=cv2.rectangle(overlay, (0, 0), (0+(len("Not following : 100"))*17, 110), (0,0,0), -1)

        img = cv2.putText(img, "Total People : {:.0f}".format(tot), (0, 30),
                        cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1)
    
        img = cv2.putText(img, "SD Index : {:.2f}%".format(sd_index), (0 , 60),
                        cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255), 1)
        
        img = cv2.putText(img, "Not Following : {:.0f}".format(count), (0 , 90),
                        cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255), 1)
        
        
        alpha = 0.5

        cv2.addWeighted(overlay, alpha, output, 1 - alpha,
        0, output)
        
        out.write(output)

        cv2.imshow("output", output)
        if cv2.waitKey(1) == ord('q'):
            break
    df_tot = pd.DataFrame(df_tot)
    print(df_tot)
    
    df_tot.to_csv(FLAGS.output_csv,index = False)
    
    
    vid.release()
    
    out.release()
        
    cv2.destroyAllWindows()



if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
