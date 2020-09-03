#!/usr/bin/python3

import os;
from os import mkdir;
from os.path import exists, join;
import tensorflow as tf;
import tensorflow _datasets as tfds;
from models import DETR, Loss;

os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1';
batch_size = 8;

def main():

  detr = DETR(80, 50);
  detr_loss = Loss(80, 50);
  trainset = tfds.load(name = 'coco2014', split = tfds.Split.TRAIN, download = False);
  trainset = trainset.map(map_function).repeat(100).shuffle(batch_size).apply(tf.data.experimental.dense_to_ragged_batch(batch_size)).prefetch(tf.data.experimental.AUTOTUNE);
  validationset = tfds.load(name = 'coco2014', split = tfds.Split.VALIDATION, download = False);
  validationset_iter = iter(validationset.map(map_function).repeat(100).shuffle(batch_size).apply(tf.data.experimental.dense_to_ragged_batch(batch_size)).prefetch(tf.data.experimental.AUTOTUNE));
  testset = tfds.load(name = 'coco2014', split = tfds.Split.TEST, download = False);
  testset = testset.repeat(100).prefetch(tf.data.experimental.AUTOTUNE);
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
  for images, labels in trainset:
    # images.shape = (batch, h, w, 3)
    # bboxes.shape = (batch, ragged target_num, 4)
    # classes.shape = (batch, ragged target_num)
    bbox_gt, labels_gt = labels;
    with tf.GradientTape() as tape:
      labels_pred, bbox_pred = detr(images);
      loss = detr_loss(bbox_pred, labels_pred, bbox_gt, labels_gt);
    train_loss.update_state(loss);
    if tf.equal(optimizer.iterations % 10, 0):
      with log.as_default():
        tf.summary.scalar('train loss', train_loss.result());
      train_loss.reset_states();
    grads = tape.gradient(loss, detr.trainable_variables);
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
      images = features['image'].numpy().astype('uint8');
      

if __name__ == "__main__":

  main();
