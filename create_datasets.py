#!/usr/bin/python3

import json;
from os import listdir;
from os.path import join, exists, isfile;
import cv2;
import tensorflow as tf;

def parse_function(serialized_example):

  feature = tf.io.parse_single_example(
    serialized_example,
    features = {
      'data': tf.io.FixedLenFeature((), dtype = tf.string, default_value = ''),
      'shape': tf.io.FixedLenFeature((3,), dtype = tf.int64),
      'bbox': tf.io.VarLenFeature(dtype = tf.float32),
      'label': tf.io.VarLenFeature(dtype = tf.int64),
      'is_crowd': tf.io.VarLenFeature(dtype = tf.int64)
    }
  );
  shape = tf.cast(feature['shape'], dtype = tf.int32);
  data = tf.io.decode_jpeg(feature['data']);
  data = tf.reshape(data, shape);
  data = tf.cast(data, dtype = tf.float32);
  bbox = tf.sparse.to_dense(feature['bbox'], default_value = 0);
  bbox = tf.reshape(bbox, (-1, 4)); # bbox = (x,y,w,h)
  bbox = tf.concat([tf.reverse(bbox[...,:2], axis = [-1]), tf.reverse(bbox[...,:2], axis = [-1]) + tf.reverse(bbox[...,2:], axis = [-1])], axis = -1) / \
    tf.cast([[shape[0], shape[1], shape[0], shape[1]]], dtype = tf.float32); # bbox = (y1, x1, y2, x2) / (height, width, height, width)
  label = tf.sparse.to_dense(feature['label'], default_value = 0);
  label = tf.reshape(label, (-1,));
  is_crowd = tf.sparse.to_dense(feature['is_crowd'], default_value = 0);
  is_crowd = tf.reshape(is_crowd, (-1,));
  return {'image': data, 'objects': {'bbox': bbox, 'label': label, 'is_crowd': is_crowd}};

def create_dataset(image_dir, label_dir, trainset = True):

  if exists(join(label_dir, 'instances_train2014.json')):
    with open(join(label_dir, 'instances_train2014.json' if trainset else 'instances_val2014.json'), 'r') as f:
      labels = json.loads(f.read());
  elif exists(join(label_dir, 'instances_train2017.json')):
    with open(join(label_dir, 'instances_train2017.json' if trainset else 'instances_val2017.json'), 'r') as f:
      labels = json.loads(f.read());
  else:
    print('can\'t find annotation file!');
    exit(1);
  annotations = dict();
  # 1) collect images
  for image in labels['images']:
    img_id = image['id'];
    img_path = join(image_dir, image['file_name']);
    if exists(img_path) == False:
      print('can\'t read image ' + img_path);
      continue;
    if img_id not in annotations:
      annotations[img_id] = {'path': img_path,
                             'bbox': tf.zeros((0,4), dtype = tf.float32), 
                             'label': tf.zeros((0,), dtype = tf.int32), 
                             'is_crowd': tf.zeros((0,), dtype = tf.int32)};
  # 2) collect annotations
  for annotation in labels['annotations']:
    img_id = annotation['image_id'];
    if img_id not in annotations:
      print('image id %d not found' % (img_id));
      continue;
    bbox = tf.expand_dims(tf.constant(annotation['bbox'], dtype = tf.float32), axis = 0); # bbox.shape = (1,4)
    category_id = tf.expand_dims(tf.constant(annotation['category_id'], dtype = tf.int32), axis = 0); # category_id.shape = (1,)
    is_crowd = tf.expand_dims(tf.constant(annotation['iscrowd'], dtype = tf.int32), axis = 0); # is_crowd.shape = (1,)
    annotations[img_id]['bbox'] = tf.concat([annotations[img_id]['bbox'], bbox], axis = 0); # bbox.shape = (n, 4)
    annotations[img_id]['label'] = tf.concat([annotations[img_id]['label'], category_id], axis = 0); # label.shape = (n,)
    annotations[img_id]['is_crowd'] = tf.concat([annotations[img_id]['is_crowd'], is_crowd], axis = 0); # is_crowd.shape = (n,)
  # 3) generate tfrecord
  writer = tf.io.TFRecordWriter('trainset.tfrecord' if trainset else 'testset.tfrecord');
  if writer is None:
    print('invalid output file!');
    exit(1);
  for anno in annotations.values():
    img = cv2.imread(anno['path']);
    if img is None:
      print('can\'t read image %s' % (anno['path']));
      continue;
    trainsample = tf.train.Example(features = tf.train.Features(
      feature = {
        'data': tf.train.Feature(bytes_list = tf.train.BytesList(value = [tf.io.encode_jpeg(img).numpy()])),
        'shape': tf.train.Feature(int64_list = tf.train.Int64List(value = list(img.shape))),
        'bbox': tf.train.Feature(float_list = tf.train.FloatList(value = tf.reshape(anno['bbox'], (-1,)))),
        'label': tf.train.Feature(int64_list = tf.train.Int64List(value = anno['label'])),
        'is_crowd': tf.train.Feature(int64_list = tf.train.Int64List(value = anno['is_crowd']))
      }
    ));
    writer.write(trainsample.SerializeToString());
  writer.close();

if __name__ == "__main__":

  assert tf.executing_eagerly() == True;
  from sys import argv;
  if len(argv) != 4:
    print('Usage: %s <train image dir>  <test image dir> <anno dir>' % (argv[0],));
    exit(1);
  create_dataset(argv[1], argv[3], True);
  create_dataset(argv[2], argv[3], False);
