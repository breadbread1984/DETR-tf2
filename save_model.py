#!/usr/bin/python3

import tensorflow as tf;
from models import DETR;

def main():

  detr = DETR(80, 50);
  optimizer = tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.ExponentialDecay(1e-4, decay_steps = 110000, decay_rate = 0.99));
  checkpoint = tf.train.Checkpoint(model = detr, optimizer = optimizer);
  checkpoint.restore(tf.train.latest_checkpoint('checkpoints'));
  detr.save('detr.h5');
  detr.save_weights('detr_weights.h5');

if __name__ == "__main__":

  assert tf.executing_eagerly();
  main();
