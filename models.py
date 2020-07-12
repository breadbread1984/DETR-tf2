#!/usr/bin/python3

from math import pi;
import numpy as np;
import tensorflow as tf;
from Transformer import EncoderLayer, DecoderLayer;

def PositionEmbeddingSine(hidden_dim = 64, normalize = True, eps = 1e-6):

  d_model = hidden_dim // 2;
  inputs = tf.keras.Input((None, None, hidden_dim)); # inputs.shape = (batch, height, width , d_model)
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

  # NOTE: the learned positional embedding can only support image as large as 1600 x 1600
  d_model = hidden_dim // 2;
  inputs = tf.keras.Input((None, None, hidden_dim));
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
  # 2) input with positional embedding
  if position_embedding == 'sine':
    embeddings = PositionEmbeddingSine(d_model)(inputs); # embeddings.shape = (batch, height, width, d_model)
  elif position_embedding == 'learned':
    embeddings = PositionEmbeddingLearned(d_model)(inputs); # embeddings.shape = (batch, height, width, d_model)
  else: raise Exception('unknonw position embedding!');
  embeddings = tf.keras.layers.Reshape((-1, d_model))(embeddings); # outputs.shape = (batch, height * width, d_model)
  mask_flatten = tf.keras.layers.Reshape((1,1,-1))(mask); # mask.shape = (batch, 1, 1(will be height * width), height * width)
  outputs = tf.keras.layers.Dropout(rate = dropout_rate)(embeddings);  # embeddings.shape = (batch, encode_length, d_model)
  # 3) multiple encode layers
  for i in range(num_layers):
    outputs = EncoderLayer(d_model, num_heads, code_dim, dropout_rate, activation)([outputs, mask_flatten]); # outputs.shape = (batch, encode_length, d_model)
  outputs = tf.keras.layers.Lambda(lambda x: tf.reshape(x[0], tf.shape(x[1])))([outputs, inputs]); # outputs.shape = (batch, height, width, d_model)
  return tf.keras.Model(inputs = (inputs, mask), outputs = outputs);

def ImageDecoder(num_layers, d_model, num_heads, code_dim, dropout_rate, activation = 'relu', position_embedding = 'sine'):
    
  assert activation in ['relu', 'gelu'];
  # d_model must be divisible by num_heads.
  tf.debugging.Assert(tf.equal(d_model % num_heads,0),[d_model, num_heads]);
  # 1) inputs
  inputs = tf.keras.Input((None, d_model));          # inputs.shape = (batch, num_queries, d_model)
  code = tf.keras.Input((None, None, d_model));      # code.shape = (batch, height, width, d_model)
  padding_mask = tf.keras.Input((1, 1, None));       # padding_mask.shape = (batch, 1, 1, num_queries)
  # 2) code with positional embedding
  # NOTE: original Transformer doesnt add positional embedding to code
  if position_embedding == 'sine':
    code_with_pos = PositionEmbeddingSine(d_model)(code); # code_with_pos.shape = (batch, height, width, d_model)
  elif position_embedding == 'learned':
    code_with_pos = PositionEmbeddingLearned(d_model)(code); # code_with_pos.shape = (batch, height, width, d_model)
  else: raise Exception('unknonw position embedding!');
  code_with_pos = tf.keras.layers.Reshape((-1, d_model))(code_with_pos); # outputs.shape = (batch, height * width, d_model)
  look_ahead_mask = tf.keras.layers.Lambda(lambda x: tf.zeros((tf.shape(x[0])[0], 1, tf.shape(x[1])[1], tf.shape(x[1])[1]), dtype = tf.float32))([code_with_pos, inputs]); # look_ahead_mask.shape = (batch, 1, num_queries, height * width)
  outputs = tf.keras.layers.Dropout(rate = dropout_rate)(inputs); # outputs.shape = (batch, decode_length, d_model)
  # 3) multiple decode layers
  for i in range(num_layers):
    outputs = DecoderLayer(d_model, num_heads, code_dim, dropout_rate, activation)([outputs, code_with_pos, look_ahead_mask, padding_mask]); # outputs.shape = (batch, decode_length, d_model)
  return tf.keras.Model(inputs = (inputs, code, padding_mask), outputs = outputs);

def ImageTransformer(num_classes, num_layers = 2, num_queries = 100, d_model = 256, num_heads = 8, code_dim = 2048, dropout_rate = 0.1, activation = 'relu', position_embedding = 'sine'):
    
  # NOTE: num_queries is the number of object the model outputs
  assert activation in ['relu', 'gelu'];
  # 1) inputs
  inputs = tf.keras.Input((None, None, d_model));                                                                                                          # inputs.shape = (batch, height, width, d_model)
  queries = tf.keras.layers.Lambda(lambda x, n: tf.tile(tf.expand_dims(tf.range(tf.cast(n, dtype = tf.float32)), axis = 0), (tf.shape(x)[0], 1)), arguments = {'n': num_queries})(inputs); # queries.shape = (batch, num_queries)
  dec_inputs = tf.keras.layers.Embedding(num_queries, d_model)(queries);                                                                                   # dec_inputs.shape = (batch, num_queries, d_model)
  enc_padding_mask = tf.keras.layers.Lambda(lambda x: tf.zeros(tf.shape(x)[0:3], dtype = tf.float32))(inputs);                                             # enc_padding_mask.shape = (batch, height, width)
  dec_padding_mask = tf.keras.layers.Lambda(lambda x: tf.zeros((tf.shape(x)[0], 1, 1, tf.shape(x)[1] * tf.shape(x)[2]), dtype = tf.float32))(inputs);      # dec_padding_mask.shape = (batch, 1, 1(will be num_queries), height * width)
  # 2) generate code
  code = ImageEncoder(num_layers, d_model, num_heads, code_dim, dropout_rate, activation, position_embedding)([inputs, enc_padding_mask]);              # code.shape = (batch, height, width, d_model)
  decoded = ImageDecoder(num_layers, d_model, num_heads, code_dim, dropout_rate, activation, position_embedding)([dec_inputs, code, dec_padding_mask]); # decoded.shape = (batch, num_queries, d_model)
  # 3) output
  # predict class
  classes = tf.keras.layers.Dense(units = num_classes + 1, activation = tf.keras.layers.Softmax())(decoded); # outputs.shape = (batch, num_queries, num_classes + 1)
  # predict coordinates
  results = tf.keras.layers.Dense(units = d_model, activation = tf.keras.layers.ReLU())(decoded);
  results = tf.keras.layers.Dense(units = d_model, activation = tf.keras.layers.ReLU())(results);
  coords = tf.keras.layers.Dense(units = 4, activation = tf.math.sigmoid)(results); # coord.shape = (batch, num_queries, 4)
  return tf.keras.Model(inputs = inputs, outputs = (classes, coords));

def DETR(num_classes, target_num = 100, num_layers = 6, hidden_dim = 256, code_dim = 2048, position_embedding = 'sine'):

  assert position_embedding in ['sine', 'learned'];
  inputs = tf.keras.Input((None, None, 3)); # inputs.shape = (batch, height, width, 3)
  resnet50 = tf.keras.applications.ResNet50(input_tensor = inputs, include_top = False, weights = 'imagenet');
  results = tf.keras.layers.Conv2D(filters = hidden_dim, kernel_size = (1, 1), padding = 'same')(resnet50.get_layer('conv5_block3_out').output);
  classes, coords = ImageTransformer(num_classes, num_layers = num_layers, num_queries = target_num, d_model = hidden_dim, code_dim = code_dim, position_embedding = position_embedding)(results);
  return tf.keras.Model(inputs = inputs, outputs = (classes, coords));

def HungarianMatcher(num_classes, pos_weight = 1., iou_weight = 1., class_weight = 1.):

  bbox_pred = tf.keras.Input((None, 4)); # bbox_pred.shape = (batch = 1, num_queries, 4)
  labels_pred = tf.keras.Input((None, num_classes + 1)); # labels_pred.shape = (batch = 1, num_queries, num_classes + 1)
  bbox_gt = tf.keras.Input((None, 4)); # bbox_gt.shape = (batch = 1, num_targets, 4)
  labels_gt = tf.keras.Input((None, )); # labels_gt.shape = (batch = 1, num_targets)
  # 1) get 1-norm of box prediction
  bbox_pred_reshape = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis = -2))(bbox_pred); # bbox_pred_reshape.shape = (batch, num_queries, 1, 4)
  bbox_gt_reshape = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis = -3))(bbox_gt); # bbox_gt_reshape.shape = (batch, 1, num_targets, 4)
  bbox_loss = tf.keras.layers.Lambda(lambda x: tf.norm(x[0] - x[1], ord = 1, axis = -1))([bbox_pred_reshape, bbox_gt_reshape]); # bbox_loss.shape = (batch, num_queries, num_targets)
  # 2) get iou = intersect / (area_a + area_b - intersect)
  bbox_pred_ul = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x[..., 0:2] - 0.5 * x[..., 2:4], axis = -2))(bbox_pred); # bbox_pred_ul.shape = (batch, num_queries, 1, 2)
  bbox_pred_dr = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x[..., 0:2] + 0.5 * x[..., 2:4], axis = -2))(bbox_pred); # bbox_pred_dr.shape = (batch, num_queries, 1, 2)
  bbox_gt_ul = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x[..., 0:2] - 0.5 * x[..., 2:4], axis = -3))(bbox_gt); # bbox_gt_ul.shape = (batch, 1, num_targets, 2)
  bbox_gt_dr = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x[..., 0:2] + 0.5 * x[..., 2:4], axis = -3))(bbox_gt); # bbox_gt_dr.shape = (batch, 1, num_targets, 2)
  upperleft = tf.keras.layers.Lambda(lambda x: tf.math.maximum(x[0], x[1]))([bbox_pred_ul, bbox_gt_ul]); # upperleft.shape = (batch, num_queries, num_targets, 2)
  downright = tf.keras.layers.Lambda(lambda x: tf.math.minimum(x[0], x[1]))([bbox_pred_dr, bbox_gt_dr]); # downright.shape = (batch, num_queries, num_targets, 2)
  intersect_wh = tf.keras.layers.Lambda(lambda x: tf.math.maximum(x[1] - x[0] + 1, 0.))([upperleft, downright]); # intersect_wh.shape = (batch, num_queries, num_targets, 2)
  intersect_area = tf.keras.layers.Lambda(lambda x: x[...,0] * x[...,1])(intersect_wh); # intersect_area.shape = (batch,num_queries, num_targets)
  bbox_pred_wh = tf.keras.layers.Lambda(lambda x: tf.math.maximum(x[1] - x[0] + 1, 0.))([bbox_pred_ul, bbox_pred_dr]); # bbox_pred_wh.shape = (batch, num_queries, 1, 2)
  bbox_pred_area = tf.keras.layers.Lambda(lambda x: x[...,0] * x[...,1])(bbox_pred_wh); # bbox_pred_area.shape = (batch, num_queries, 1)
  bbox_gt_wh = tf.keras.layers.Lambda(lambda x: tf.math.maximum(x[1] - x[0] + 1, 0.))([bbox_gt_ul, bbox_gt_dr]); # bbox_gt_wh.shape = (batch, 1, num_targets, 2)
  bbox_gt_area = tf.keras.layers,Lambda(lambda x: x[...,0] * x[...,1])(bbox_gt_wh); # bbox_gt_area.shape = (batch, 1, num_targets)
  iou = tf.keras.layers.Lambda(lambda x: x[0] / (x[1] + x[2] - x[0]))([intersect_area, bbox_pred_area, bbox_gt_area]); # iou.shape = (batch, num_queries, num_targets)
  # 3) get bg_ratio = [bounding - (area_a + area_b - intersect)] / bounding
  upperleft = tf.keras.layers.Lambda(lambda x: tf.math.minimum(x[0], x[1]))([bbox_pred_ul, bbox_gt_ul]); # upperleft.shape = (batch, num_queries, num_targets, 2)
  downright = tf.keras.layers.Lambda(lambda x: tf.math.maximum(x[0], x[1]))([bbox_pred_dr, bbox_gt_dr]); # downright.shape = (batch, num_queries, num_targets, 2)
  bounding_wh = tf.keras.layers.Lambda(lambda x: tf.math.maximum(x[1] - x[0] + 1, 0.))([upperleft, downright]); # intersect_wh.shape = (batch, num_queries, num_targets, 2)
  bounding_area = tf.keras.layers.Lambda(lambda x: x[...,0] * x[...,1])(bounding_wh); # bounding_area.shape = (batch, num_queries, num_targets)
  bg_ratio = tf.keras.layers.Lambda(lambda x: (x[0] - (x[2] + x[3] - x[1])) / x[0])([bounding_area, intersect_area, bbox_pred_area, bbox_gt_area]); # bg_ratio.shape = (batch, num_queries, num_targets)
  # 4) get iou loss
  iou_loss = tf.keras.layers.Lambda(lambda x: -(x[0] - x[1]))([iou, bg_ratio]); # iou_loss.shape = (batch, num_queries, num_targets)
  # 5) get class loss
  def fn(x):
    labels_pred_slice = x[0]; # labels_pred_slice.shape = (num_queries, num_classes + 1)
    labels_gt_slice = x[1]; # labels_gt_slices.shape = (num_targets)
    y = tf.reshape(tf.range(tf.cast(tf.shape(labels_pred_slice)[0], dtype = tf.float32)), (-1, 1, 1)); # y.shape = (num_queries, 1, 1)
    x = tf.reshape(labels_gt_slice, (1, -1, 1)); # x.shape = (1, num_targets, 1)
    yx = tf.concat([y,x], axis = -1); # yx.shape = (num_queries, num_targets, 2)
    values = tf.gather_nd(labels_pred_slice, yx); # values.shape = (num_queries, num_targets)
    return values;
  class_loss = tf.keras.layers.Lambda(lambda x: -tf.map_fn(fn, (x[0], x[1])))([labels_pred, labels_gt]); # class_loss.shape = (batch, num_queries, num_targets)
  # 6) sum
  loss = tf.keras.layers.Lambda(lambda x, p, i, c: p * x[0] + i * x[1] + c * x[2], arguments = {'p': pos_weight, 'i': iou_weight, 'c': class_weight})([bbox_loss, iou_loss, class_loss]); # loss.shape = (batch, num_queries, num_targets)
  

if __name__ == "__main__":

  assert tf.executing_eagerly();  
  detr = DETR(50);
  a = tf.constant(np.random.normal(size = (8, 480, 640, 3)), dtype = tf.float32);
  classes, coords = detr(a);
  print(classes.shape, coords.shape)
  detr.save('detr.h5');
  tf.keras.utils.plot_model(model = detr, to_file = 'DETR.png', show_shapes = True, dpi = 64);
