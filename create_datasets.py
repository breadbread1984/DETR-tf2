#!/usr/bin/python3

from os import mkdir, listdir;
from os.path import join, exists;
from shutil import rmtree;
from math import ceil;
from multiprocessing import Process;
from pycocotools.coco import COCO;
import numpy as np;
import cv2;
import tensorflow as tf;

PROCESS_NUM = 80;
label_map = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, -1, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, -1, 25, 26, -1, -1, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, -1, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, -1, 61, -1, -1, 62, -1, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, -1, 74, 75, 76, 77, 78, 79, 80];

def parse_function_generator(num_classes, img_shape = (416,416), random = True, jitter = .3):
  def parse_function(serialized_example):
    feature = tf.io.parse_single_example(
      serialized_example,
      features = {
        'image': tf.io.FixedLenFeature((), dtype = tf.string),
        'bbox': tf.io.VarLenFeature(dtype = tf.float32),
        'label': tf.io.VarLenFeature(dtype = tf.int64),
        'obj_num': tf.io.FixedLenFeature((), dtype = tf.int64)
      });
    obj_num = tf.keras.layers.Lambda(lambda x: tf.cast(x, dtype = tf.int32))(feature['obj_num']);
    image = tf.keras.layers.Lambda(lambda x: tf.io.decode_jpeg(x))(feature['image']);
    bbox = tf.keras.layers.Lambda(lambda x: tf.sparse.to_dense(x, default_value = 0))(feature['bbox']);
    bbox = tf.keras.layers.Lambda(lambda x: tf.reshape(x[0], (x[1], 4)))([bbox, obj_num]);
    labels = tf.keras.layers.Lambda(lambda x: tf.sparse.to_dense(x, default_value = 0))(feature['label']);
    labels = tf.keras.layers.Lambda(lambda x: tf.reshape(x[0], (x[1],)))([labels, obj_num]); # labels.shape = (obj_num)
    # add batch dimension
    image = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis = 0))(image);
    bbox = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis = 0))(bbox);
    # augmentation
    if random == True:
      aspect_ratio_jitter = tf.keras.layers.Lambda(lambda x, j: tf.random.uniform(shape = (2,), minval = 1 - j, maxval = 1 + j, dtype = tf.float32), arguments = {'j': jitter})(image); # aspect_ratio_jitter.shape = (2)
      resize_input_shape = tf.keras.layers.Lambda(lambda x, h, w: tf.cast([h, w], dtype = tf.float32) * x, arguments = {'h': img_shape[1], 'w': img_shape[0]})(aspect_ratio_jitter); # resize_input_shape.shape = (2) in sequence of (h, w)
      scale = tf.keras.layers.Lambda(lambda x: tf.random.uniform(shape = (1,), minval = .8, maxval = 1.2, dtype = tf.float32))(image); # scale.shape = (1)
      resize_shape = tf.keras.layers.Lambda(lambda x: tf.cast(tf.cond(tf.greater(x[1][0], x[1][1]), true_fn = lambda: x[2] * x[1] / x[0][0], false_fn = lambda: x[2] * x[1] / x[0][1]), dtype = tf.int32))([aspect_ratio_jitter, resize_input_shape, scale]); # resize_shape.shape = (2) in sequence of (h, w)
      resize_image = tf.keras.layers.Lambda(lambda x: tf.image.resize(x[0], x[1], method = tf.image.ResizeMethod.BICUBIC))([image, resize_shape]);
      # 1) try to pad along height direction
      pad = tf.keras.layers.Lambda(lambda x, h: tf.math.maximum(h - x[0], 0), arguments = {'h': img_shape[1]})(resize_shape);
      pad_top = tf.keras.layers.Lambda(lambda x: tf.random.uniform(maxval = x + 1, shape = (), dtype = tf.int32))(pad);
      pad_bottom = tf.keras.layers.Lambda(lambda x: x[0] - x[1])([pad, pad_top]);
      hpad_image = tf.keras.layers.Lambda(lambda x: tf.pad(x[0], [[0,0],[x[1],x[2]],[0,0],[0,0]], constant_values = 128))([resize_image, pad_top, pad_bottom]);
      hpad_bbox = tf.keras.layers.Lambda(lambda x: x[0] * tf.cast([[x[1][0], x[1][1], x[1][0], x[1][1]]], dtype = tf.float32))([bbox, resize_shape]);
      hpad_bbox = tf.keras.layers.Lambda(lambda x: x[0] + tf.cast([[x[1], 0, x[1], 0]], dtype = tf.float32))([hpad_bbox, pad_top]);
      hpad_bbox = tf.keras.layers.Lambda(lambda x: x[0] / tf.cast([[x[1][0] + x[2], x[1][1], x[1][0] + x[2], x[1][1]]], dtype = tf.float32))([hpad_bbox, resize_shape, pad]);
      resize_shape = tf.keras.layers.Lambda(lambda x: x[0] + tf.cast([x[1], 0], dtype = tf.int32))([resize_shape, pad]); # resize_shape = (h + pad, w)
      # 2) try to calculate crop along height direction
      crop = tf.keras.layers.Lambda(lambda x, h: tf.math.maximum(x[0] - h, 0), arguments = {'h': img_shape[1]})(resize_shape);
      crop_top = tf.keras.layers.Lambda(lambda x: tf.random.uniform(maxval = x + 1, shape = (), dtype = tf.int32))(crop);
      hcrop_image = tf.keras.layers.Lambda(lambda x, h: tf.image.crop_to_bounding_box(x[0], x[1], 0, h, tf.shape(x[0])[2]), arguments = {'h': img_shape[1]})([hpad_image, crop_top]);
      hcrop_bbox = tf.keras.layers.Lambda(lambda x: x[0] * tf.cast([[x[1][0], x[1][1], x[1][0], x[1][1]]], dtype = tf.float32))([hpad_bbox, resize_shape]);
      hcrop_bbox = tf.keras.layers.Lambda(lambda x: x[0] - tf.cast([[x[1], 0, x[1], 0]], dtype = tf.float32))([hcrop_bbox, crop_top]);
      hcrop_bbox = tf.keras.layers.Lambda(lambda x: x[0] / tf.cast([[x[1][0] - x[2], x[1][1], x[1][0] - x[2], x[1][1]]], dtype = tf.float32))([hcrop_bbox, resize_shape, crop]);
      resize_shape = tf.keras.layers.Lambda(lambda x: x[0] - tf.cast([x[1], 0], dtype = tf.int32))([resize_shape, crop]); # resize_shape = (h - crop, w)
      # 3) try to pad along width direction
      pad = tf.keras.layers.Lambda(lambda x, w: tf.math.maximum(w - x[1], 0), arguments = {'w': img_shape[0]})(resize_shape);
      pad_left = tf.keras.layers.Lambda(lambda x: tf.random.uniform(maxval = x + 1, shape = (), dtype = tf.int32))(pad);
      pad_right = tf.keras.layers.Lambda(lambda x: x[0] - x[1])([pad, pad_left]);
      wpad_image = tf.keras.layers.Lambda(lambda x: tf.pad(x[0], [[0,0],[0,0],[x[1],x[2]],[0,0]], constant_values = 128))([hcrop_image, pad_left, pad_right]);
      wpad_bbox = tf.keras.layers.Lambda(lambda x: x[0] * tf.cast([[x[1][0], x[1][1], x[1][0], x[1][1]]], dtype = tf.float32))([hcrop_bbox, resize_shape]);
      wpad_bbox = tf.keras.layers.Lambda(lambda x: x[0] + tf.cast([[0, x[1], 0, x[1]]], dtype = tf.float32))([wpad_bbox, pad_left]);
      wpad_bbox = tf.keras.layers.Lambda(lambda x: x[0] / tf.cast([[x[1][0], x[1][1] + x[2], x[1][0], x[1][1] + x[2]]], dtype = tf.float32))([wpad_bbox, resize_shape, pad]);
      resize_shape = tf.keras.layers.Lambda(lambda x: x[0] + tf.cast([0, x[1]], dtype = tf.int32))([resize_shape, pad]); # resize_shape = (h, w + pad)
      # 4) try to calculate crop along width direction
      crop = tf.keras.layers.Lambda(lambda x, w: tf.math.maximum(x[1] - w, 0), arguments = {'w': img_shape[0]})(resize_shape);
      crop_left = tf.keras.layers.Lambda(lambda x: tf.random.uniform(maxval = x + 1, shape = (), dtype = tf.int32))(crop);
      wcrop_image = tf.keras.layers.Lambda(lambda x, w: tf.image.crop_to_bounding_box(x[0], 0, x[1], tf.shape(x[0])[1], w), arguments = {'w': img_shape[0]})([wpad_image, crop_left]);
      wcrop_bbox = tf.keras.layers.Lambda(lambda x: x[0] * tf.cast([[x[1][0], x[1][1], x[1][0], x[1][1]]], dtype = tf.float32))([wpad_bbox, resize_shape]);
      wcrop_bbox = tf.keras.layers.Lambda(lambda x: x[0] - tf.cast([[0, x[1], 0, x[1]]], dtype = tf.float32))([wcrop_bbox, crop_left]);
      wcrop_bbox = tf.keras.layers.Lambda(lambda x: x[0] / tf.cast([[x[1][0], x[1][1] - x[2], x[1][0], x[1][1] - x[2]]], dtype = tf.float32))([wcrop_bbox, resize_shape, crop]);
      resize_shape = tf.keras.layers.Lambda(lambda x: x[0] - tf.cast([0, x[1]], dtype = tf.int32))([resize_shape, crop]); # resize_shape = (h, w - crop)
      # 5) random flip image
      flip = tf.keras.layers.Lambda(lambda x: tf.math.less(np.random.rand(), 0.5))(image);
      flip_image = tf.keras.layers.Lambda(lambda x: tf.cond(x[1], true_fn = lambda: tf.image.flip_left_right(x[0]), false_fn = lambda: x[0]))([wcrop_image, flip]);
      final_bbox = tf.keras.layers.Lambda(lambda x: tf.cond(x[1], true_fn = lambda: x[0] * tf.cast([1,-1,1,-1], dtype = tf.float32) + tf.cast([0,1,0,1], dtype = tf.float32), false_fn = lambda: x[0]))([wcrop_bbox, flip]);
      # 6) distort image in HSV color space
      color_distort_image = tf.keras.layers.Lambda(lambda x: tf.image.random_hue(x, 10 / 180))(flip_image);
      color_distort_image = tf.keras.layers.Lambda(lambda x: tf.image.random_saturation(x, 0, 10))(color_distort_image);
      final_image = tf.keras.layers.Lambda(lambda x: tf.image.random_brightness(x, 10 / 255))(color_distort_image);
    else:
      resize_image = tf.keras.layers.Lambda(lambda x, h, w: tf.image.resize(x, (h, w), method = tf.image.ResizeMethod.BICUBIC, preserve_aspect_ratio = True), arguments = {'h': img_shape[1], 'w': img_shape[0]})(image); # resize_image.shape = (batch, nh, nw, 3)
      pad_image = tf.keras.layers.Lambda(lambda x, h, w: tf.pad(x, [[0,0],
                                                                    [(h - tf.shape(x)[1])//2, (h - tf.shape(x)[1]) - (h - tf.shape(x)[1])//2],
                                                                    [(w - tf.shape(x)[2])//2, (w - tf.shape(x)[2]) - (w - tf.shape(x)[2])//2],
                                                                    [0,0]], constant_values = 128), 
                                        arguments = {'h': img_shape[1], 'w': img_shape[0]})(resize_image); # resize_image.shape = (batch, 416, 416, 3)
      final_image = tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float32))(pad_image); # image_data.shape = (batch, 416, 416, 3)
      resize_bbox = tf.keras.layers.Lambda(lambda x: x[0] * tf.cast([[[tf.shape(x[1])[1], 
                                                                      tf.shape(x[1])[2], 
                                                                      tf.shape(x[1])[1], 
                                                                      tf.shape(x[1])[2]]]], dtype = tf.float32))([bbox, resize_image]); # resize_bbox.shape = (batch, obj_num, 4)
      pad_bbox = tf.keras.layers.Lambda(lambda x, h, w: x[0] + tf.cast([[[(h - tf.shape(x[1])[1])//2,
                                                                          (w - tf.shape(x[1])[2])//2,
                                                                          (h - tf.shape(x[1])[1])//2,
                                                                          (w - tf.shape(x[1])[2])//2]]], dtype = tf.float32), 
                                        arguments = {'h': img_shape[1], 'w': img_shape[0]})([resize_bbox, resize_image]);
      final_bbox = tf.keras.layers.Lambda(lambda x, h, w: x / tf.cast([[[h, w, h, w]]], dtype = tf.float32),
                                          arguments = {'h': img_shape[1], 'w': img_shape[0]})(pad_bbox);
    final_image = tf.keras.layers.Lambda(lambda x: x / 255.)(final_image);
    image = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis = 0))(final_image); # image.shape = (height, width, 3)
    bbox = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis = 0))(final_bbox); # bbox.shape = (obj_num, 4)
    return image, (bbox, labels);
  return parse_function;

def parse_function(serialized_example):
  
  feature = tf.io.parse_single_example(
    serialized_example,
    features = {
      'image': tf.io.FixedLenFeature((), dtype = tf.string),
      'bbox': tf.io.VarLenFeature(dtype = tf.float32),
      'label': tf.io.VarLenFeature(dtype = tf.int64),
      'obj_num': tf.io.FixedLenFeature((), dtype = tf.int64)
    });
  obj_num = tf.cast(feature['obj_num'], dtype = tf.int32);
  image = tf.io.decode_jpeg(feature['image']);
  bbox = tf.sparse.to_dense(feature['bbox'], default_value = 0);
  bbox = tf.reshape(bbox, (obj_num, 4));
  label = tf.sparse.to_dense(feature['label'], default_value = 0);
  label = tf.reshape(label, [obj_num]);
  return image, bbox, label;

def create_dataset(image_dir, label_dir, trainset = True):

  anno = COCO(join(label_dir, 'instances_train2017.json' if trainset else 'instances_val2017.json'));
  if exists('trainset' if trainset else 'testset'): rmtree('trainset' if trainset else 'testset');
  mkdir('trainset' if trainset else 'testset');
  imgs_for_each = ceil(len(anno.getImgIds()) / PROCESS_NUM);
  handlers = list();
  filenames = list();
  for i in range(PROCESS_NUM):
    filename = ('trainset_part_%d' if trainset else 'testset_part_%d') % i;
    filenames.append(join('trainset' if trainset else 'testset', filename));
    handlers.append(Process(target = worker, args = (join('trainset' if trainset else 'testset', filename), anno, image_dir, anno.getImgIds()[i * imgs_for_each:(i+1) * imgs_for_each] if i != PROCESS_NUM - 1 else anno.getImgIds()[i * imgs_for_each:])));
    handlers[-1].start();
  for handler in handlers:
    handler.join();

def worker(filename, anno, image_dir, image_ids):
  writer = tf.io.TFRecordWriter(filename);
  for image in image_ids:
    img_info = anno.loadImgs([image])[0];
    height, width = img_info['height'], img_info['width'];
    img = cv2.imread(join(image_dir, img_info['file_name']));
    if img is None:
      print('can\'t open image %s' % (join(image_dir, img_info['file_name'])));
      continue;
    annIds = anno.getAnnIds(imgIds = image);
    anns = anno.loadAnns(annIds);
    bboxs = list();
    labels = list();
    for ann in anns:
      # bounding box
      bbox_x, bbox_y, bbox_w, bbox_h = ann['bbox'];
      # relative upper left y, x, bottom right y, x with respect to the height and width
      bbox = tf.constant([bbox_y / height, bbox_x / width, (bbox_y + bbox_h) / height, (bbox_x + bbox_w) / width], dtype = tf.float32);
      bboxs.append(bbox);
      # category
      category = label_map[ann['category_id']];
      assert category != -1 and category != 0; # can't be not presented category (-1) can't be background category (0)
      labels.append(category - 1);
      assert 0 <= labels[-1] <= 79;
    obj_num = len(bboxs);
    bboxs = tf.cast(tf.stack(bboxs, axis = 0), dtype = tf.float32); # bboxs.shape = (obj_num, 4)
    labels = tf.cast(tf.stack(labels, axis = 0), dtype = tf.int32); # labels.shape = (obj_num)
    assert labels.shape[0] == bboxs.shape[0] and labels.shape[0] == obj_num;
    trainsample = tf.train.Example(features = tf.train.Features(
      feature = {
        'image': tf.train.Feature(bytes_list = tf.train.BytesList(value = [tf.io.encode_jpeg(img).numpy()])),
        'bbox': tf.train.Feature(float_list = tf.train.FloatList(value = tf.reshape(bboxs, (-1)))),
        'label': tf.train.Feature(int64_list = tf.train.Int64List(value = tf.reshape(labels, (-1)))),
        'obj_num': tf.train.Feature(int64_list = tf.train.Int64List(value = [obj_num]))
      }
    ));
    writer.write(trainsample.SerializeToString());
  writer.close();

if __name__ == "__main__":

  '''
  # this code is for testing data augmentation
  trainset_filenames = [join('testset', filename) for filename in listdir('testset')];
  trainset = tf.data.TFRecordDataset(trainset_filenames).map(parse_function_generator(80));
  for image, labels in trainset:
    image = image * 255.; # image.shape = (416, 416, 3)
    labels1, labels2, labels3 = labels; # labels1.shape = (13, 13, 3, 85) labels2.shape = (26, 26, 3, 85) labels3.shape = (52, 52, 3, 85)
    mask1 = tf.math.equal(labels1[..., 4], 1); # mask1.shape = (13, 13, 3)
    mask2 = tf.math.equal(labels2[..., 4], 1); # mask2.shape = (26, 26, 3)
    mask3 = tf.math.equal(labels3[..., 4], 1); # msak3.shape = (52, 52, 3)
    labels1 = tf.boolean_mask(labels1, mask1); # labels1.shape = (obj_num, 85)
    labels2 = tf.boolean_mask(labels2, mask2); # labels2.shape = (obj_num, 85)
    labels3 = tf.boolean_mask(labels3, mask3); # labels3.shape = (obj_num, 85)
    bbox = tf.concat([labels1[..., 0:4], labels2[..., 0:4], labels3[..., 0:4]], axis = 0); # bbox.shape = (total obj num, 4) in sequence of (center x, y, w, h)
    half_wh = bbox[..., 2:4] / 2;
    upperleft = (bbox[..., 0:2] - half_wh) * tf.cast([tf.shape(image)[1], tf.shape(image)[0]], dtype = tf.float32); # upperleft.shape = (total obj num, 2)
    bottomright = (bbox[..., 0:2] + half_wh) * tf.cast([tf.shape(image)[1], tf.shape(image)[0]], dtype = tf.float32); # bottomright.shape = (total obj num, 2)
    bbox = tf.concat([upperleft, bottomright], axis = -1); # bbox.shape = (total obj num, 4)
    img = image.numpy().astype('uint8');
    for box in bbox:
      ul = tuple(box[0:2].numpy().astype(int));
      br = tuple(box[2:4].numpy().astype(int));
      cv2.rectangle(img, ul, br, (0,255,0), 2);
    cv2.imshow('objects', img);
    cv2.waitKey();
  exit(0);
  '''

  from sys import argv;
  if len(argv) != 4:
    print("Usage: " + argv[0] + "<train image dir> <test image dir> <anno dir>");
    exit(1);
  assert tf.executing_eagerly() == True;
  create_dataset(argv[2], argv[3], False);
  create_dataset(argv[1], argv[3], True);
