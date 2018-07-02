#-*- coding:utf-8 -*-
import xml.etree.ElementTree as ET
import numpy as np
import os
import tensorflow as tf
from PIL import Image
from os import getcwd

wd = getcwd() #get the path of current project
# sets = [('2007', 'trainval'), ('2012', 'trainval')]
dirname = wd+'/data/our-cut-two/'
xml_file_name = wd+'/data/our-cut-two/trainval.txt'

sets = [('our', 'trainval')]

classes = ["bus", "car", "truck"]

# return the scale size of object boubdingbox to [0,1] with respecet to the whole box
def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return [x, y, w, h]


def convert_annotation(image_id):
    in_file = open('%s/data/our-cut-two/Annotations/%s.xml'%(wd,image_id),encoding='utf8')

    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    bboxes = []
    for i, obj in enumerate(root.iter('object')):
        if i > 29:
            break
        # difficult = obj.find('difficult').text
        cls = obj.find('name').text
        # if cls not in classes or int(difficult) == 1:
        if cls not in classes:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w, h), b) + [cls_id]
        bboxes.extend(bb)
    if len(bboxes) < 30*5:
        bboxes = bboxes + [0, 0, 0, 0, 0]*(30-int(len(bboxes)/5))

    return np.array(bboxes, dtype=np.float32).flatten().tolist()

def convert_img(image_id):
    image = Image.open('%s/data/our-cut-two/Original/%s.jpg' % (wd, image_id))
    resized_image = image.resize((416, 416), Image.BICUBIC)
    image_data = np.array(resized_image, dtype='float32')/255
    img_raw = image_data.tobytes()
    return img_raw

# filename = os.path.join('trainval'+'0712'+'.tfrecords')
filename = os.path.join('trainval'+'_our-cut-two'+'.tfrecords')
writer = tf.python_io.TFRecordWriter(filename)

def reName(dirname,xml_file_name):

    file_handle = open(xml_file_name, mode='w+')
    flag = False
    # if not os.path.exists(filename_path):
    #     os.mknod(filename_path)
    # print(os.listdir(dirname))
    # print(os.listdir(dirname)[-2])
    for category in os.listdir(dirname):
        # print(category)
        catdir = os.path.join(dirname,category)
        if not os.path.isdir(catdir):
            continue
        if category == os.listdir(dirname)[-2]:
            flag = True
        # print(flag)
        files = os.listdir(catdir)
        # print(files)
        # files.remove('.DS_Store')
        count = 0
        for cur_file in files:
            print("handle" + category + " " + cur_file)
            filename = os.path.join(catdir,cur_file)
            count = count + 1
            oldDir = os.path.join(catdir,cur_file)
            if os.path.isdir(oldDir):
                continue
            filename=os.path.splitext(cur_file)[0]
            filetype=os.path.splitext(cur_file)[1]
            # newDir=os.path.join(catdir,str(count)+filetype)
            # newDir=os.path.join(catdir,str((6-len(str(count))))*'0'+str(count)+filetype)
            newDir=os.path.join(catdir,str(count).zfill(6)+filetype)
            os.rename(oldDir,newDir)
            #write to file
            if flag:
                file_handle.write(str(count).zfill(6)+'\n')
    file_handle.close()


reName(dirname,xml_file_name)
# for year, image_set in sets:
image_ids = open(xml_file_name).read().strip().split()
# print(filename)
for image_id in image_ids:
    xywhc = convert_annotation(image_id)
    img_raw = convert_img(image_id)

    example = tf.train.Example(features=tf.train.Features(feature={
        'xywhc':
                tf.train.Feature(float_list=tf.train.FloatList(value=xywhc)),
        'img':
                tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
        }))
    writer.write(example.SerializeToString())
writer.close()
