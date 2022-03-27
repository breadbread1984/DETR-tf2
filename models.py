#!/usr/bin/python3

from math import pi;
import numpy as np;
from scipy.optimize import linear_sum_assignment;
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

def ImageTransformer(num_classes, num_layers = 6, num_queries = 100, d_model = 512, num_heads = 8, code_dim = 2048, dropout_rate = 0.1, activation = 'relu', position_embedding = 'sine'):
    
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

  '''
  # NOTE: the following code is not supported by tensorflow 2.3
  # 7) assign num_targets assignments to num_queries workers
  def assign_py(cost):
    # cost.shape = (num_queries, num_targets) and num_targets < num_queries
    row_ind, col_ind = linear_sum_assignment(cost); # row_ind.shape = (num_targets) col_ind.shape = (num_targets)
    return row_ind, col_ind;
  @tf.function
  def assign(cost):
    row_ind, col_ind = tf.py_function(assign_py, inp = [cost], Tout = tf.int64);
    ind = tf.stack([row_ind,col_ind], axis = -1); # ind.shape = (num_targets, 2)
    ind = tf.cast(ind, dtype = tf.int32);
    return ind;
  ind = tf.keras.layers.Lambda(lambda x: tf.map_fn(assign, x))(cost);
  return tf.keras.Model(inputs = (bbox_pred, labels_pred, bbox_gt, labels_gt), outputs = ind);
  '''

class Loss(tf.keras.Model):

  def __init__(self, num_classes = 100, target_num = 100, weights = {'label_loss': 1, 'bbox_loss': 5, 'iou_loss': 2, 'cardinality_loss': 1}):
    super(Loss, self).__init__();
    self._weights = weights;
  def combination_losses(self, _bbox_pred, _labels_pred, _bbox_gt, _labels_gt):
    costs = list();
    for x in zip(_bbox_pred, _labels_pred, _bbox_gt, _labels_gt):
      bbox_pred = x[0]; # bbox_pred.shape = (num_queries, 4)
      labels_pred = x[1]; # labels_pred.shape = (num_queries, num_classes + 1)
      bbox_gt = x[2]; # bbox_gt_slice.shape = (num_targets, 4)
      labels_gt = x[3]; # labels_gt_slice.shape = (num_targets)
      # 1) get 1-norm of box prediction
      bbox_pred = tf.expand_dims(bbox_pred, axis = -2); # bbox_pred.shape = (num_queries, 1, 4)
      bbox_gt = tf.expand_dims(bbox_gt, axis = -3); # bbox_gt.shape = (1, num_targets, 4)
      bbox_loss = tf.norm(bbox_pred - bbox_gt, ord = 1, axis = -1); # bbox_loss.shape = (num_queries, num_targets)
      # get iou = intersect / (area_a + area_b - intersect)
      bbox_pred_ul = bbox_pred[..., 0:2] - 0.5 * bbox_pred[..., 2:4]; # bbox_pred_ul.shape = (num_queries, 1, 2)
      bbox_pred_dr = bbox_pred[..., 0:2] + 0.5 * bbox_pred[..., 2:4]; # bbox_pred_dr.shape = (num_queries, 1, 2)
      bbox_gt_ul = bbox_gt[..., 0:2] - 0.5 * bbox_gt[..., 2:4]; # bbox_gt_ul.shape = (1, num_targets, 2)
      bbox_gt_dr = bbox_gt[..., 0:2] + 0.5 * bbox_gt[..., 2:4]; # bbox_gt_dr.shape = (1, num_targets, 2)
      upperleft = tf.math.maximum(bbox_pred_ul, bbox_gt_ul); # upperleft.shape = (num_queries, num_targets, 2)
      downright = tf.math.minimum(bbox_pred_dr, bbox_gt_dr); # downright.shape = (num_queries, num_targets, 2)
      intersect_wh = tf.math.maximum(downright - upperleft + 1, 0.); # intersect_wh.shape = (num_queries, num_targets, 2)
      intersect_area = intersect_wh[...,0] * intersect_wh[...,1]; # intersect_area.shape = (num_queries, num_targets)
      bbox_pred_wh = tf.math.maximum(bbox_pred_dr - bbox_pred_ul + 1, 0.); # bbox_pred_wh.shape = (num_queries, 1, 2)
      bbox_pred_area = bbox_pred_wh[...,0] * bbox_pred_wh[...,1]; # bbox_pred_area.shape = (num_queries, 1)
      bbox_gt_wh = tf.math.maximum(bbox_gt_dr - bbox_gt_ul + 1, 0.); # bbox_gt_wh.shape = (1, num_targets, 2)
      bbox_gt_area = bbox_gt_wh[...,0] * bbox_gt_wh[...,1]; # bbox_gt_area.shape = (1, num_targets)
      iou = intersect_area / tf.math.maximum(bbox_pred_area + bbox_gt_area - intersect_area, 1e-9); # iou.shape = (num_queries, num_targets)
      # get bg_ratio = [bounding - (area_a + area_b - intersect)] / bounding
      upperleft = tf.math.minimum(bbox_pred_ul, bbox_gt_ul); # upperleft.shape = (num_queries, num_targets, 2)
      downright = tf.math.maximum(bbox_pred_dr, bbox_gt_dr); # downright.shape = (num_queries, num_targets, 2)
      bounding_wh = tf.math.maximum(downright - upperleft + 1, 0.); # intersect_wh.shape = (num_queries, num_targets, 2)
      bounding_area = bounding_wh[...,0] * bounding_wh[...,1]; # bounding_area.shape = (num_queries, num_targets)
      bg_ratio = (bounding_area - (bbox_pred_area + bbox_gt_area - intersect_area)) / tf.math.maximum(bounding_area, 1e-9); # bg_ratio.shape = (num_queries, num_targets)
      # 2) get giou loss
      iou_loss = -(iou - bg_ratio); # iou_loss.shape = (num_queries, num_targets)
      # 3) get class loss
      probs = tf.gather(labels_pred, labels_gt, axis = -1); # probs.shape = (num_queries, num_targets)
      class_loss = -probs; # loss.shape = (num_queries, num_targets)
      # 4) sum
      cost = bbox_loss + iou_loss + class_loss; # loss.shape = (num_queries, num_targets)
      costs.append(cost);
    return costs;
  def call(self, x):
    bbox_pred = x[0]; # bbox_pred.shape = (batch, num_queries, 4)
    labels_pred = x[1]; # labels_pred.shape = (batch, num_queries, num_classes + 1)
    bbox_gt = x[2]; # bbox_gt.shape = (batch, ragged num_targets, 4)
    labels_gt = x[3]; # labels_gt.shape = (batch, ragged num_targets)
    # 1) match detections and groundtruths
    costs = self.combination_losses(bbox_pred, labels_pred, bbox_gt, labels_gt); # costs.shape = (batch, num queries, ragged num_targets)
    assignments = list();
    for cost in costs:
      # row_ind: which detection
      # col_ind: which ground truth
      row_ind, col_ind = linear_sum_assignment(cost.numpy());
      assignment = tf.stack([row_ind, col_ind], axis = -1);
      assignment = tf.cast(assignment, dtype = tf.int32); # assignment.shape = (num_targets, 2) in sequence of detection_id->ground truth_id
      assignments.append(assignment);
    # assignments.shape = (batch, ragged num_targets, 2) in sequence of (query idx, target idx)
    # 2) label loss
    label_losses = list();
    for pred, gt, assignment in zip(labels_pred, labels_gt, assignments):
      # NOTE: labels = 0 represents no object
      query_idx = tf.expand_dims(assignment[...,0], axis = -1); # query_idx.shape = (target_num, 1)
      target_labels = tf.gather(gt, assignment[..., 1]); # target_labels.shape = (target_num)
      shape = tf.constant([pred.shape[0]]); # shape = [num_queries]
      query_labels = tf.scatter_nd(query_idx, target_labels, shape); # query_labels.shape = (num_queries) with unassigned query target to class 0
      # NOTE: pred is output of softmax already
      loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = False)(query_labels, pred);
      label_losses.append(loss);
    # 3) cardinality loss
    cardinality_losses = list();
    for pred, gt in zip(labels_pred, labels_gt):
      classes = tf.math.argmax(pred, axis = -1); # classes.shape = (num_queries)
      pred_obj_num = tf.math.reduce_sum(tf.cast(tf.math.not_equal(classes, 0), dtype = tf.float32)); # obj_num.shape = ()
      gt_obj_num = tf.cast(gt.shape[0], dtype = tf.float32); # gt_obj_num.shape = ()
      loss = tf.math.abs(gt_obj_num - pred_obj_num);
      cardinality_losses.append(loss);
    # 4) boxes loss
    bbox_losses = list();
    iou_losses = list();
    for pred, gt, assignment in zip(bbox_pred, bbox_gt, assignments):
      if gt.shape[0] == 0:
        # no target labeled for this image
        bbox_losses.append(0);
        iou_losses.append(0);
        continue;
      pred = tf.gather(pred, assignment[...,0]); # pred.shape = (target_num, 4)
      target = tf.gather(gt, assignment[...,1]); # target.shape = (target_num, 4)
      # regression loss
      reg_loss = tf.keras.losses.MeanAbsoluteError()(target, pred); # loss.shape = ()
      # iou loss
      bbox_pred_ul = pred[...,0:2] - 0.5 * pred[...,2:4]; # bbox_pred_ul.shape = (num_targets, 2)
      bbox_pred_dr = pred[...,0:2] + 0.5 * pred[...,2:4]; # bbox_pred_dr.shape = (num_targets, 2)
      bbox_gt_ul = gt[...,0:2] - 0.5 * gt[...,2:4]; # bbox_gt_ul.shape = (num_targets, 2)
      bbox_gt_dr = gt[...,0:2] + 0.5 * gt[...,2:4]; # bbox_gt_dr.shape = (num_targets, 2)
      intersect_ul = tf.math.maximum(bbox_pred_ul, bbox_gt_ul); # intersect_ul.shape = (num_targets, 2)
      intersect_dr = tf.math.minimum(bbox_pred_dr, bbox_gt_dr); # intersect_dr.shape = (num_targets, 2)
      intersect_wh = tf.math.maximum(intersect_dr - intersect_ul + 1, 0); # intersect_wh.shape = (num_targets, 2)
      intersect_area = intersect_wh[...,0] * intersect_wh[...,1]; # intersect_area.shape = (num_targets)
      bounding_ul = tf.math.minimum(bbox_pred_ul, bbox_gt_ul); # bounding_ul.shape = (num_targets, 2)
      bounding_dr = tf.math.maximum(bbox_pred_dr, bbox_gt_dr); # bounding_dr.shape = (num_targets, 2)
      bounding_wh = tf.math.maximum(bounding_dr - bounding_ul + 1, 0); # bounding_wh.shape = (num_targets, 2)
      bounding_area = bounding_wh[...,0] * bounding_wh[...,1]; # bounding_area.shape = (num_targets)
      bbox_pred_wh = tf.math.maximum(bbox_pred_dr - bbox_pred_ul + 1, 0); # bbox_pred_wh.shape = (num_targets, 2)
      bbox_pred_area = bbox_pred_wh[...,0] * bbox_pred_wh[...,1]; # bbox_pred_area.shape = (num_targets)
      bbox_gt_wh = tf.math.maximum(bbox_gt_dr - bbox_gt_ul + 1, 0); # bbox_gt_wh.shape = (num_targets, 2)
      bbox_gt_area = bbox_gt_wh[...,0] * bbox_gt_wh[...,1]; # bbox_gt_area.shape = (num_targets)
      union = bbox_pred_area + bbox_gt_area - intersect_area; # union.shape = (num_targets)
      iou = intersect_area / tf.math.maximum(union, 1e-9); # iou.shape = (num_targets)
      bg_ratio = (bounding_area - union) / tf.math.maximum(bounding_area, 1e-9); # bg_ratio.shape = (num_targets)
      iou_loss = tf.math.reduce_mean(iou - bg_ratio); # iou_loss.shape = ()
      bbox_losses.append(reg_loss);
      iou_losses.append(iou_loss);
    # 5) sum up losses
    loss = tf.stack(label_losses) * self._weights['label_loss'] + tf.stack(bbox_losses) * self._weights['bbox_loss'] + tf.stack(iou_losses) * self._weights['iou_loss'];
    loss = tf.math.reduce_sum(loss); # loss.shape = ()
    return loss;

if __name__ == "__main__":

  assert tf.executing_eagerly();  
  detr = DETR(50);
  a = tf.constant(np.random.normal(size = (8, 480, 640, 3)), dtype = tf.float32);
  classes, coords = detr(a);
  print(classes.shape, coords.shape)
  detr.save('detr.h5');
  #tf.keras.utils.plot_model(model = detr, to_file = 'DETR.png', show_shapes = True, dpi = 64);
