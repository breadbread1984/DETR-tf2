#!/usr/bin/python3

import os;
from os import mkdir;
from os.path import exists, join;
import numpy as np;
import cv2;
import tensorflow as tf;
from create_datasets import parse_function;
from preprocess import map_function;
from models import DETR, Loss;
from predictor import Predictor;

os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1';
batch_size = 8;

def main():

  detr = DETR(80, 50);
  detr_loss = Loss(80, 50);
  trainset = tf.data.TFRecordDataset('coco14/trainset.tfrecord').repeat(100).map(parse_function).map(map_function).apply(tf.data.experimental.dense_to_ragged_batch(batch_size));
  validationset = tf.data.TFRecordDataset('coco14/testset.tfrecord').repeat(100).map(parse_function).map(map_function).apply(tf.data.experimental.dense_to_ragged_batch(batch_size));
  validationset_iter = iter(validationset);
  testset = tf.data.TFRecordDataset('coco14/testset.tfrecord').repeat(100).map(parse_function).prefetch(tf.data.experimental.AUTOTUNE);
  testset_iter = iter(testset);
  # restore from existing checkpoint
  optimizer = tf.keras.optimizers.Adam(1e-4);
  if False == exists('checkpoints'): mkdir('checkpoints');
  checkpoint = tf.train.Checkpoint(model = detr, optimizer = optimizer);
  checkpoint.restore(tf.train.latest_checkpoint('checkpoints'));
  # tensorboard summary
  log = tf.summary.create_file_writer('checkpoints');
  # train model
  train_loss = tf.keras.metrics.Mean(name = 'train_loss', dtype = tf.float32);
  validation_loss = tf.keras.metrics.Mean(name = 'validation_loss', dtype = tf.float32);
  for images, bbox_gt, labels_gt in trainset:
    # images.shape = (batch, h, w, 3)
    # bboxes.shape = (batch, ragged target_num, 4)
    # classes.shape = (batch, ragged target_num)
    with tf.GradientTape() as tape:
      labels_pred, bbox_pred = detr(images);
      loss = detr_loss(bbox_pred, labels_pred, bbox_gt, labels_gt);
    train_loss.update_state(loss);
    if tf.equal(optimizer.iterations % 10, 0):
      with log.as_default():
        tf.summary.scalar('train loss', train_loss.result());
      train_loss.reset_states();
    grads = tape.gradient(loss, detr.trainable_variables);
    optimizer.apply_gradients(zip(grads, detr.trainable_variables));
    if tf.equal(optimizer.iterations % 2000, 0):
      # save checkpoint every 1000 steps
      checkpoint.save(join('checkpoints', 'ckpt'));
      detr.save('detr.h5');
    if tf.equal(optimizer.iterations % 100, 0):
      print('validating on validation set...');
      for i in range(10):
        images, labels = next(validationset_iter);
        bbox_gt, labels_gt = labels;
        labels_pred, bbox_pred = detr(images);
        loss = detr_loss(bbox_pred, labels_pred, bbox_gt, labels_gt);
        validation_loss.update_state(loss);
      with log.as_default():
        tf.summary.scalar('validation loss', validation_loss.result(), step = optimizer.iterations);
      validation_loss.reset_states();
      print('testing on test set...');
      features = next(testset_iter);
      image = features['image'].numpy().astype('uint8');
      predictor = Predictor(detr = detr);
      boundings = predictor.predict(image);
      color_map = dict();
      for bounding in boundings:
        if bounding[5].numpy().astype('int32') in color_map:
          clr = color_map[bounding[5].numpy().astype('int32')];
        else:
          color_map[bounding[5].numpy().astype('int32')] = tuyple(np.random.randint(low=0, high=256,size=(3,)).tolist());
          clr = color_map[bounding[5].numpy().astype('int32')];
        cv2.rectangle(image, tuple(bounding[0:2].numpy().astype('int32')), tuple(bounding[2:4].numpy().astype('int32')), clr, 5);
      image = tf.expand_dims(img, axis = 0);
      with log.as_default():
        tf.summary.image('detect', img, step = optimizer.iterations);
  detr.save('detr.h5');

if __name__ == "__main__":

  assert tf.executing_eagerly();
  main();
