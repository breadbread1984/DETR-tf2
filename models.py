#!/usr/bin/python3

from math import pi;
import numpy as np;
import tensorflow as tf;

def PositionEmbeddingSine(hidden_dim = 64, normalize = True, eps = 1e-6):

  d_model = hidden_dim // 2;
  inputs = tf.keras.Input((None, None, d_model)); # inputs.shape = (batch, height, width , d_model)
  y_positions = tf.keras.layers.Lambda(lambda x: tf.tile(tf.reshape(tf.range(tf.cast(tf.shape(x)[1], dtype = tf.float32)), (-1,1,1)), (1,tf.shape(x)[2],1)))(inputs); # y_positions.shape = (height, width, 1)
  x_positions = tf.keras.layers.Lambda(lambda x: tf.tile(tf.reshape(tf.range(tf.cast(tf.shape(x)[2], dtype = tf.float32)), (1,-1,1)), (tf.shape(x)[1],1,1)))(inputs); # x_positions.shape = (height, width, 1)
  if normalize:
    y_positions = tf.keras.layers.Lambda(lambda x, e, p: x / (x[-1:, :, :] + eps) * 2 * p, arguments = {'e': eps, 'p': pi})(y_positions);
    x_positions = tf.keras.layers.Lambda(lambda x, e, p: x / (x[:, -1:, :] + eps) * 2 * p, arguments = {'e': eps, 'p': pi})(x_positions);
  j = tf.keras.layers.Lambda(lambda x, d: tf.tile(tf.reshape(tf.range(tf.cast(d, dtype = tf.float32)), (1, 1, -1)), (tf.shape(x)[1], tf.shape(x)[2], 1)), arguments = {'d': d_model})(inputs); # j.shape = (height, width, d_model)
  i = tf.keras.layers.Lambda(lambda x: x // 2)(j); # i.shape = (height, width, d_model)
  power = tf.keras.layers.Lambda(lambda x, d: 2 * x / tf.cast(d, dtype = tf.float32), arguments = {'d': d_model})(i); # power.shape = (height, width, d_model)
  y_angles = tf.keras.layers.Lambda(lambda x: x[0] / tf.math.pow(10000., x[1]))([y_positions, power]); # y_angles.shape = (height, width, d_model)
  x_angles = tf.keras.layers.Lambda(lambda x: x[0] / tf.math.pow(10000., x[1]))([x_positions, power]); # x_angles.shape = (height, width, d_model)
  y_sines = tf.keras.layers.Lambda(lambda x: tf.math.sin(x[:,:,0::2]))(y_angles); # y_sines.shape = (height, width, d_model // 2)
  x_sines = tf.keras.layers.Lambda(lambda x: tf.math.sin(x[:,:,0::2]))(x_angles); # x_sines.shape = (height, width, d_model // 2)
  y_cosines = tf.keras.layers.Lambda(lambda x: tf.math.cos(x[:,:,1::2]))(y_angles); # y_cosines.shape = (height, width, d_model // 2)
  x_cosines = tf.keras.layers.Lambda(lambda x: tf.math.cos(x[:,:,1::2]))(x_angles); # x_cosines.shape = (height, width, d_model // 2)
  pos_encoding = tf.keras.layers.Concatenate()([y_sines, y_cosines, x_sines, x_cosines]); # pos_encoding.shape = (height, width, d_model * 2)
  pos_encoding = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis = 0))(pos_encoding); # pos_encoding.shape = (1, height, width, hidden_dim)
  return tf.keras.Model(inputs = inputs, outputs = pos_encoding);

def PositionEmbeddingLearned(hidden_dim = 256):

  d_model = hidden_dim // 2;
  inputs = tf.keras.Input((None, None, d_model));
  y_positions = tf.keras.layers.Lambda(lambda x: tf.expand_dims(tf.range(tf.cast(tf.shape(x)[1], dtype = tf.float32)), axis = 0))(inputs); # y_positions.shape = (1, height)
  x_positions = tf.keras.layers.Lambda(lambda x: tf.expand_dims(tf.range(tf.cast(tf.shape(x)[2], dtype = tf.float32)), axis = 0))(inputs); # x_positions.shape = (1, width)
  y_embedding = tf.keras.layers.Embedding(50, d_model)(y_positions); # y_embedding.shape = (1, height, d_model)
  x_embedding = tf.keras.layers.Embedding(50, d_model)(x_positions); # x_embedding.shape = (1, width, d_model)
  y_mesh = tf.keras.layers.Lambda(lambda x: tf.tile(tf.reshape(x[0], (tf.shape(x[0])[1], 1, tf.shape(x[0])[2])), (1, tf.shape(x[1])[1], 1)))([y_embedding, x_embedding]); # y_mesh.shape = (height, width, d_model)
  x_mesh = tf.keras.layers.Lambda(lambda x: tf.tile(tf.reshape(x[0], (1, tf.shape(x[0])[1], tf.shape(x[0])[2])), (tf.shape(x[1])[1], 1, 1)))([x_embedding, y_embedding]); # x_mesh.shape = (height, widht, d_model)
  pos_encoding = tf.keras.layers.Concatenate(axis = -1)([y_mesh, x_mesh]); # pos_encoding.shape = (height, width, 2 * d_model)
  pos_encoding = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis = 0))(pos_encoding); # pos_encoding.shape = (1, height, width, hidden_dim)
  return tf.keras.Model(inputs = inputs, outputs = pos_encoding);

def CNN(position_embedding = 'sine'):

  assert position_embedding in ['sine', 'learned'];
  inputs = tf.keras.Input((None, None, 3)); # inputs.shape = (batch, height, width, 3)
  if position_embedding == 'sine':
    pass;
  elif position_embedding == 'learned':
    pass;
  else: raise Exception('unknonw position embedding!');
  resnet50 = tf.keras.applications.ResNet50(input_tensor = inputs, include_top = False, weights = 'imagenet');
  resnet50.get_layer('conv5_block3_out').output
  print(resnet50.get_layer('conv5_block3_out').output.shape)

if __name__ == "__main__":

  assert tf.executing_eagerly();
  a = tf.constant(np.random.normal(size = (8, 10, 20, 2048)));
  b = PositionEmbeddingSine()(a);
  print(b.shape)
  b = PositionEmbeddingLearned()(a);
  print(b.shape)
