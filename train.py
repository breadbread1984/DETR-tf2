#!/usr/bin/python3

import os;
from os import mkdir;
from os.path import exists, join;
import numpy as np;
import cv2;
import tensorflow as tf;
from create_datasets import parse_function_generator;
from models import DETR, Loss;
from predictor import Predictor;

os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1';
batch_size = 8;

def main():

  detr = DETR(80, 50);
  detr_loss = Loss(80, 50);
  optimizer = tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.ExponentialDecay(1e-4, decay_steps = 110000, decay_rate = 0.99));
  checkpoint = tf.train.Checkpoint(model = detr, optimizer = optimizer);
  train_loss = tf.keras.metrics.Mean(name = 'train loss', dtype = tf.float32);
  test_loss = tf.keras.metrics.Mean(name = 'test loss', dtype = tf.float32);
  # load downloaded dataset
  trainset_filenames = [join('trainset', filename) for filename in listdir('trainset')];
  testset_filenames = [join('testset', filename) for filename in listdir('testset')];
  trainset = tf.data.TFRecordDataset(trainset_filenames).map(parse_function_generator(80)).repeat(-1).apply(tf.data.experimental.dense_to_ragged_batch(batch_size));
  testset = tf.data.TFRecordDataset(testset_filenames).map(parse_function_generator(80)).repeat(-1).apply(tf.data.experimental.dense_to_ragged_batch(batch_size));
  validationset = tf.data.TFRecordDataset(testset_filenames).map(parse_function).repeat(-1);
  trainset_iter = iter(trainset);
  testset_iter = iter(testset);
  validationset_iter = iter(validationset);
  # restore from existing checkpoint
  if False == exists('checkpoints'): mkdir('checkpoints');
  checkpoint.restore(tf.train.latest_checkpoint('checkpoints'));
  # tensorboard summary
  log = tf.summary.create_file_writer('checkpoints');
  # train model
  while True:
    images, (bbox, labels) = next(trainset_iter);
    with tf.GradientTape() as tape:
      labels_pred, bbox_pred = detr(images);
      loss = detr_loss(bbox_pred, labels_pred, bbox, labels);
    # check whether the loss numeric is correct
    if tf.math.reduce_any(tf.math.is_nan(loss)) == True:
      print("NaN was detected in loss, skip the following steps!");
      continue;
    grads = tape.gradient(loss, detr.trainable_variables);
    # check whether the grad numeric is correct
    if tf.math.reduce_any([tf.math.reduce_any(tf.math.is_nan(grad)) for grad in grads]) == True:
      print("NaN was detected in gradients, skip gradient apply!");
      continue;
    optimizer.apply_gradients(zip(grads, detr.trainable_variables));
    train_loss.update_state(loss);
    # save model
    if tf.equal(optimizer.iterations % 10000, 0):
      # save checkpoint every 1000 steps
      checkpoint.save(join('checkpoints', 'ckpt'));
      detr.save('detr.h5');
    if tf.equal(optimizer.iterations % 100, 0):
      # evaluate
      for i in range(10):
        images, (bbox, labels) = next(testset_iter);
        labels_pred, bbox_pred = detr(images);
        loss = detr_loss(bbox_pred, labels_pred, bbox, labels);
        test_loss.update_state(loss);
      # visualize
      image, bbox, labels = next(validationset_iter);
      predictor = Predictor(detr = detr);
      img = image.numpy().astype('uint8');
      boundings = predictor.predict(img);
      color_map = dict();
      for bounding in boundings:
        if bounding[5].numpy().astype('int32') in color_map:
          clr = color_map[bounding[5].numpy().astype('int32')];
        else:
          color_map[bounding[5].numpy().astype('int32')] = tuyple(np.random.randint(low=0, high=256,size=(3,)).tolist());
          clr = color_map[bounding[5].numpy().astype('int32')];
        cv2.rectangle(img, tuple(bounding[0:2].numpy().astype('int32')), tuple(bounding[2:4].numpy().astype('int32')), clr, 5);
      img = tf.expand_dims(img, axis = 0);
      # write log
      with log.as_default():
        tf.summary.scalar('train loss', train_loss.result(), step = optimizer.iterations);
        tf.summary.scalar('test loss', test_loss.result(), step = optimizer.iterations);
        tf.summary.image('detect', img[...,::-1], step = optimizer.iterations);
      print('Step #%d Train Loss: %.6f Test Loss: %.6f' % (optimizer.iterations, train_loss.result(), test_loss.result()));
      # break condition
      #if train_loss.result() < 0.001: break;
      # reset
      train_loss.reset_states();
      test_loss.reset_states();
  detr.save('detr.h5');

if __name__ == "__main__":

  assert tf.executing_eagerly();
  main();
