#!/usr/bin/python3

from os.path import exists;
import numpy as np;
import tensorflow as tf;
from models import DETR;

class Predictor(object):

  def __init__(self, query_num = 50, class_num = 80, input_shape = (416, 416, 3), detr = None):

    # load model object from serialized file or assign directly
    self.input_shape = input_shape;
    if detr is None:
      if exists('detr.h5'):
        self.detr = tf.keras.models.load_model('detr.h5', compile = False);
      else:
        self.detr = DETR(class_num, query_num);
        optimizer = tf.keras.optimizers.Adam(1e-4);
        checkpoint = tf.train.Checkpoint(model = self.detr, optimizer = optimizer);
        checkpoint.restore(tf.train.latest_checkpoint('checkpoints'));
    else:
      self.detr = detr;

  def predict(self, image):

    # preprocess image
    images = tf.expand_dims(image, axis = 0);
    resize_images = tf.image.resize(images, self.input_shape[:2], method = tf.image.ResizeMethod.BICUBIC, preserve_aspect_ratio = True);
    resize_shape = resize_images.shape[1:3];
    top_pad = (self.input_shape[0] - resize_shape[0]) // 2;
    bottom_pad = self.input_shape[0] - resize_shape[0] - top_pad;
    left_pad = (self.input_shape[1] - resize_shape[1]) // 2;
    right_pad = self.input_shape[1] - resize_shape[1] - left_pad;
    resize_images = tf.pad(resize_images,[[0,0], [top_pad,bottom_pad], [left_pad,right_pad], [0,0]], constant_values = 128);
    deviation = tf.constant([left_pad / self.input_shape[1], top_pad / self.input_shape[0], 0, 0], dtype = tf.float32);
    scale = tf.constant([
      self.input_shape[1] / resize_shape[1], self.input_shape[0] / resize_shape[0],
      self.input_shape[1] / resize_shape[1], self.input_shape[0] / resize_shape[0]
    ], dtype = tf.float32);
    images_data = tf.cast(resize_images, tf.float32) / 255.;
    # detection
    labels_pred, bbox_pred = self.detr(images_data); # bbox_pred.shape = (batch, query_num, 4) labels_pred.shape = (batch, query_num, num_classes + 1)
    bbox_pred = (bbox_pred - deviation) * scale * [image.shape[1], image.shape[0], image.shape[1], image.shape[0]];
    labels_pred = tf.math.argmax(labels_pred, axis = -1); # labels_pred.shape = (batch, query_num)
    mask = tf.math.not_equal(labels_pred, 0); # mask.shape = (batch, query_num)
    masked_bbox = tf.boolean_mask(bbox_pred, mask); # masked_bbox.shape = (target_num, 4)
    masked_label = tf.expand_dims(tf.boolean_mask(labels_pred, mask), axis = -1); # masked_label.shape = (target_num, 1)
    upper_left = masked_bbox[...,0:2] - 0.5 * masked_bbox[...,2:4];
    down_right = masked_bbox[...,0:2] + 0.5 * masked_bbox[...,2:4];
    boundings = tf.keras.layers.Concatenate(axis = -1)([upper_left, down_right, tf.cast(masked_label, dtype = tf.float32)]); # boundings.shape = (target_num, 5) in sequence of (x1, y1, x2, y2, class)
    return boundings;

if __name__ == "__main__":

  assert tf.executing_eagerly() == True;
  
