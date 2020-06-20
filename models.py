#!/usr/bin/python3

from math import pi;
import numpy as np;
import tensorflow as tf;
from Transformer import EncoderLayer, DecoderLayer;

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
  results = tf.keras.layers.Lambda(lambda x: x[0] + x[1])([inputs, pos_encoding]);
  return tf.keras.Model(inputs = inputs, outputs = results);

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
  results = tf.keras.layers.Lambda(lambda x: x[0] + x[1])([inputs, pos_encoding]);
  return tf.keras.Model(inputs = inputs, outputs = results);

def ImageEncoder(num_layers, d_model, num_heads, code_dim, dropout_rate, activation = "relu", position_embedding = 'sine'):

    assert activation in ['relu', 'gelu'];
    # d_model must be divisible by num_heads.
    tf.debugging.Assert(tf.equal(d_model % num_heads,0),[d_model, num_heads]);
    # 1) inputs
    inputs = tf.keras.Input((None, None, d_model)); # inputs.shape = (batch, height, width, d_model)
    mask = tf.keras.Input((None, None));  # mask.shape = (batch, height, width)
    # 2) token to positional embedding
    if position_embedding == 'sine':
      embeddings = PositionEmbeddingSine(d_model)(inputs); # embeddings.shape = (batch, )
    elif position_embedding == 'learned':
      embeddings = PositionEmbeddingLearned(d_model)(inputs);
    else: raise Exception('unknonw position embedding!');
    embeddings = tf.keras.layers.Reshape((-1, d_model))(embeddings); # outputs.shape = (batch, height * width, d_model)
    mask = tf.keras.layers.Reshape((1,1,-1))(mask); # mask.shape = (batch, 1, 1(will be height * width), height * width)
    outputs = tf.keras.layers.Dropout(rate = dropout_rate)(embeddings);  # embeddings.shape = (batch, encode_length, dimension)
    # 3) multiple encode layers
    for i in range(num_layers):
        outputs = EncoderLayer(d_model, num_heads, code_dim, dropout_rate, activation)([outputs, mask]); # outputs.shape = (batch, encode_length, dimension)
    return tf.keras.Model(inputs = (inputs, mask), outputs = outputs);

def ImageDecoder(num_layers, d_model, num_heads, code_dim, dropout_rate, activation = 'relu', position_embedding = 'sine'):
    
    assert activation in ['relu', 'gelu'];
    # d_model must be divisible by num_heads.
    tf.debugging.Assert(tf.equal(d_model % num_heads,0),[d_model, num_heads]);
    # 1) inputs
    inputs = tf.keras.Input((None, None, d_model)); # inputs.shape = (batch, height, width, d_model)
    code = tf.keras.Input((None, d_model));            # code.shape = (batch, encode_length, dimension)
    padding_mask = tf.keras.Input((1, 1, None));       # padding_mask.shape = (batch, 1, 1, num_queries)
    # 2) token to positional embedding
    if position_embedding == 'sine':
      embeddings = PositionEmbeddingSine(d_model)(inputs); # embeddings.shape = (batch, )
    elif position_embedding == 'learned':
      embeddings = PositionEmbeddingLearned(d_model)(inputs);
    else: raise Exception('unknonw position embedding!');
    embeddings = tf.keras.layers.Reshape((-1, d_model))(embeddings); # outputs.shape = (batch, height * width, d_model)
    look_ahead_mask = tf.keras.layers.Lambda(lambda x: tf.zeros((tf.shape(x[0])[0], 1, tf.shape(x[1])[1], tf.shape(x[0])[1]), dtype = tf.float32))([code, embedding]); # look_ahead_mask.shape = (batch, 1, num_queries, code_length)
    outputs = tf.keras.layers.Dropout(rate = dropout_rate)(embeddings); # outputs.shape = (batch, decode_length, dimension)
    # 3) multiple decode layers
    for i in range(num_layers):
        outputs = DecoderLayer(d_model, num_heads, code_dim, dropout_rate, activation)([outputs, code, look_ahead_mask, padding_mask]); # outputs.shape = (batch, decode_length, dimension)
    return tf.keras.Model(inputs = (inputs, code, padding_mask), outputs = outputs);

def ImageTransformer(num_classes, num_layers = 2, num_queries = 100, d_model = 256, num_heads = 8, code_dim = 512, dropout_rate = 0.1, activation = 'relu', position_embedding = 'sine'):
    
    # NOTE: num_queries is the number of object the model outputs
    assert activation in ['relu', 'gelu'];
    # 1) inputs
    inputs = tf.keras.Input((None, None, d_model));                                                                                                          # inputs.shape = (batch, height, width, d_model)
    dec_inputs = tf.keras.layers.Lambda(lambda x, n, d: tf.random.uniform(shape = (n, d), minval = -0.05, maxval = 0.05), arguments = {'n': num_queries, 'd': d_model})(inputs); # dec_inputs.shape = (batch, num_queries, d_model)
    enc_padding_mask = tf.keras.layers.Lambda(lambda x: tf.zeros(tf.shape(x)[0:3], dtype = tf.float32))(inputs);                                             # enc_padding_mask.shape = (batch, height, width)
    dec_padding_mask = tf.keras.layers.Lambda(lambda x, n: tf.zeros((tf.shape(x)[0], 1, 1, n), dtype = tf.float32), arguments = {'n': num_queries})(inputs); # dec_padding_mask.shape = (batch, 1, 1, num_queries)
    # 2) generate code
    code = ImageEncoder(num_layers, d_model, num_heads, code_dim, dropout_rate, activation, position_embedding)([inputs, enc_padding_mask]); # code.shape = (batch, encode_length, dimension)
    decoded = ImageDecoder(num_heads, d_model, num_heads, code_dim, dropout_rate, activation, position_embedding)([dec_inputs, code, dec_padding_mask]); # decoded.shape = (batch, num_queries, dimension)
    # 3) output
    # predict class
    classes = tf.keras.layers.Dense(units = num_classes + 1)(decoded); # outputs.shape = (batch, num_queries, num_classes + 1)
    # predict coordinates
    results = tf.keras.layers.Dense(units = d_model, activation = tf.keras.layers.ReLU())(decoded);
    results = tf.keras.layers.Dense(units = d_model, activation = tf.keras.layers.ReLU())(results);
    coords = tf.keras.layers.Dense(units = 4, activation = tf.math.sigmoid)(results); # coord.shape = (batch, num_queries, 4)
    return tf.keras.Model(inputs = inputs, outputs = (classes, coords));

def DETR(num_classes, target_num = 100, num_layers = 6, hidden_dim = 256, position_embedding = 'sine'):

  assert position_embedding in ['sine', 'learned'];
  inputs = tf.keras.Input((None, None, 3)); # inputs.shape = (batch, height, width, 3)
  resnet50 = tf.keras.applications.ResNet50(input_tensor = inputs, include_top = False, weights = 'imagenet');
  results = tf.keras.layers.Conv2D(filters = hidden_dim, kernel_size = (1, 1), padding = 'same')(resnet50.get_layer('conv5_block3_out').output);
  classes, coords = ImageTransformer(num_classes, num_layers = num_layers, num_queries = target_num, d_model = hidden_dim, position_embedding = position_embedding)(results);
  return tf.keras.Model(inputs = inputs, outputs = (classes, coords));

if __name__ == "__main__":

  assert tf.executing_eagerly();
  a = tf.constant(np.random.normal(size = (8, 10, 20, 2048)));
  b = PositionEmbeddingSine()(a);
  print(b.shape)
  b = PositionEmbeddingLearned()(a);
  print(b.shape)
  
