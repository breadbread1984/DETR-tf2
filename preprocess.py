#!/usr/bin/python3

import numpy as np;
import tensorflow as tf;
import tensorflow_datasets as tfds;

# sample example
# is_crowd means whether one bbox bounds several objects
# {
# 'bbox': <tf.Tensor: id=538, shape=(1, 3, 4), dtype=float32, numpy=
# array([[[0.38278195, 0.34665626, 0.8226504 , 0.7704375 ],
#         [0.40229324, 0.65940624, 0.6712218 , 0.7636719 ],
#         [0.1137594 , 0.38051564, 0.33261278, 0.5337656 ]]], dtype=float32)>,
# 'label': <tf.Tensor: id=540, shape=(1, 3), dtype=int64, numpy=array([[9, 0, 9]])>, 
# 'is_crowd': <tf.Tensor: id=539, shape=(1, 3), dtype=bool, numpy=array([[False, False, False]])>
# }

def map_function(feature):

    image, bbox, label = tf.py_function(map_function_impl,inp = [feature["image"], feature["objects"]["bbox"], feature["objects"]["label"]],Tout = [tf.float32,tf.float32,tf.float32,tf.float32]);
    image = tf.reshape(image, (416,416,3));
    # image.shape = (416,416,3)
    # bbox.shape = (target_num, 4)
    # label.shape = (target_num)
    return image, (bbox, label);

def map_function_impl(image, bbox, label):

    image, bbox = preprocess(image, bbox, random = True);
    return image, bbox, label;

def preprocess(image, bbox, input_shape = (416,416), random = False, jitter = .3, hue = .1, sat = 1.5, bri = .1):

    # NOTE: input_shape is given in (input height, input width) order
    assert 3 == len(image.shape) and 3 == image.shape[-1];
    assert 0 < jitter < 1;
    assert -1 < hue < 1;
    assert 0 < sat;
    assert 0 < bri < 1;
    # add batch dimension
    image = tf.expand_dims(image, axis = 0);
    img_shape = image.shape[1:3]; #(height, width)

    if False == random:
        # scale the input image to make the wider edge fit the input shape
        # NOTE: I don't use resize_with_pad because it can only stuff zeros, but I want 128
        resize_image = tf.image.resize(image, input_shape, method = tf.image.ResizeMethod.BICUBIC, preserve_aspect_ratio = True);
        resize_shape = resize_image.shape[1:3]; #(height, width)
        top_pad = (input_shape[0] - resize_shape[0]) // 2;
        bottom_pad = input_shape[0] - resize_shape[0] - top_pad;
        left_pad = (input_shape[1] - resize_shape[1]) // 2;
        right_pad = input_shape[1] - resize_shape[1] - left_pad;
        resize_image = tf.pad(resize_image,[[0,0],[top_pad,bottom_pad],[left_pad,right_pad],[0,0]], constant_values = 128);
        # cast to float32
        image_data = tf.cast(resize_image, tf.float32) / 255.;
        # correct boxes
        bbox = bbox * tf.convert_to_tensor([resize_shape[0], resize_shape[1], resize_shape[0], resize_shape[1]], dtype = tf.float32);
        bbox = bbox + tf.convert_to_tensor([top_pad,left_pad,top_pad,left_pad], dtype = tf.float32);
        bbox = bbox / tf.convert_to_tensor([input_shape[0],input_shape[1],input_shape[0],input_shape[1]], dtype = tf.float32);
        # return
        return image_data, bbox;
    else:
        # randomly sample aspect ratio to input shape
        # resize image to the randomly sampled input shape
        aspect_ratio_jitter = tf.random.uniform(shape = [2], minval = 1-jitter, maxval = 1+jitter, dtype = tf.float32);
        resize_input_shape = tf.convert_to_tensor(input_shape, dtype = tf.float32) * aspect_ratio_jitter;
        scale = tf.random.uniform(shape=[1], minval = .8, maxval = 1.2, dtype = tf.float32);
        resize_shape = tf.cond(tf.greater(resize_input_shape[0],resize_input_shape[1]),true_fn = lambda: scale * resize_input_shape / aspect_ratio_jitter[0], false_fn = lambda: scale * resize_input_shape / aspect_ratio_jitter[1]);
        resize_shape = tf.cast(resize_shape, dtype = tf.int32);
        resize_image = tf.image.resize(image, resize_shape, method = tf.image.ResizeMethod.BICUBIC);
        if input_shape[0] > resize_shape[0]:
            pad = input_shape[0] - resize_shape[0];
            resize_image = tf.pad(resize_image,[[0,0],[pad,pad],[0,0],[0,0]], constant_values = 128);
            # sample crop offset_height
            offset_height = tf.random.uniform(maxval = pad + 1, dtype = tf.int32, shape = ());
            # correct boxes
            bbox = bbox * tf.convert_to_tensor([resize_shape[0], resize_shape[1], resize_shape[0], resize_shape[1]], dtype = tf.float32);
            bbox = bbox + tf.convert_to_tensor([pad, 0, pad, 0], dtype = tf.float32);
            resize_shape = resize_shape + tf.convert_to_tensor([2 * pad,0], dtype = tf.int32);
            bbox = bbox / tf.convert_to_tensor([resize_shape[0], resize_shape[1], resize_shape[0], resize_shape[1]], dtype = tf.float32);
        else:
            crop = resize_shape[0] - input_shape[0];
            # sample crop offset_height
            offset_height = tf.random.uniform(maxval = crop + 1, dtype = tf.int32, shape = ());
        if input_shape[1] > resize_shape[1]:
            pad = input_shape[1] - resize_shape[1];
            resize_image = tf.pad(resize_image,[[0,0],[0,0],[pad,pad],[0,0]], constant_values = 128);
            # sample crop offset_width
            offset_width = tf.random.uniform(maxval = pad + 1, dtype = tf.int32, shape = ());
            # correct boxes
            bbox = bbox * tf.convert_to_tensor([resize_shape[0], resize_shape[1], resize_shape[0], resize_shape[1]], dtype = tf.float32);
            bbox = bbox + tf.convert_to_tensor([0, pad, 0, pad], dtype = tf.float32);
            resize_shape = resize_shape + tf.convert_to_tensor([0, 2 * pad], dtype = tf.int32);
            bbox = bbox / tf.convert_to_tensor([resize_shape[0], resize_shape[1], resize_shape[0], resize_shape[1]], dtype = tf.float32);
        else:
            crop = resize_shape[1] - input_shape[1];
            # sample crop offset_width
            offset_width = tf.random.uniform(maxval = crop + 1, dtype = tf.int32, shape = ());
        # crop
        resize_image = tf.image.crop_to_bounding_box(resize_image, offset_height, offset_width, input_shape[0], input_shape[1]);
        # correct boxes
        bbox = bbox * tf.convert_to_tensor([resize_shape[0], resize_shape[1], resize_shape[0], resize_shape[1]], dtype = tf.float32);
        bbox = bbox + tf.convert_to_tensor([-offset_height, -offset_width, -offset_height, -offset_width], dtype = tf.float32);
        bbox = bbox / tf.convert_to_tensor([input_shape[0], input_shape[1], input_shape[0], input_shape[1]], dtype = tf.float32);
        # randomly flip image
        if np.random.rand() < .5:
            resize_image = tf.image.flip_left_right(resize_image);
            # correct boxes(y remains while x = 1 - x)
            bbox = tf.convert_to_tensor([0, 1, 0, 1], dtype = tf.float32) + tf.convert_to_tensor([1,-1,1,-1], dtype = tf.float32) * bbox;
        # distort image in HSV color space
        image_data = tf.cast(resize_image, tf.float32) / 255.;
        image_data = tf.image.random_hue(image_data, hue);
        image_data = tf.image.random_saturation(image_data, lower = 1./sat, upper = sat);
        image_data = tf.image.random_brightness(image_data, bri);
        # discard invalid boxes (small box or box having negative width or height)
        bbox = tf.clip_by_value(bbox,0,1); # restrict the min and max coordinates
        bbox_hw = bbox[...,2:4] - bbox[...,0:2] # bbox_hw.shape = (bbox_num,2)
        bbox_hw = bbox_hw * tf.convert_to_tensor(input_shape, dtype = tf.float32);
        valid = tf.math.logical_and(bbox_hw[...,0] > 1,bbox_hw[...,1] > 1); # valid.shape = (bbox_num)
        valid_bbox = tf.boolean_mask(bbox, valid); # valid_bbox.shape = (valid box num, 4)
        assert(valid_bbox.shape[1] != 0);
        # return
        return tf.squeeze(image_data), bbox;

if __name__ == "__main__":

    assert tf.executing_eagerly();
    test_num = 8;
    trainset = tfds.load(name = "coco2014", split = tfds.Split.TRAIN, download = False);
    import cv2;
    # 1) test function preprocess
    print("test function preprocess");
    count = 0;
    for feature in trainset:
        print("test the %dth image" % (count+1));
        image = feature["image"];
        bbox = feature["objects"]["bbox"];
        label = feature["objects"]["label"];
        image, bbox = preprocess(image, bbox, random = True);
        img = (image.numpy() * 255.).astype('uint8');
        for box in bbox.numpy():
            box = (box * 416).astype('int32');
            cv2.rectangle(img, (box[1],box[0]), (box[3],box[2]), (0,255,0), 1);
        cv2.imshow('img',img[...,::-1]);
        cv2.waitKey();
        count = count + 1;
        if count == test_num: break;
    print('if you can see the picture and the bounding boxes at the right places, that means the function preprocess is OK!');

    # 2) test function bbox_to_tensor
    print("test function bbox_to_tensor");
    count = 0;
    for feature in trainset:
        print("test the %dth image" % (count+1));
        image = feature["image"];
        bbox = feature["objects"]["bbox"];
        label = feature["objects"]["label"];
        image, bbox = preprocess(image, bbox, random = True);
        label1,label2,label3 = bbox_to_tensor(bbox, label);
        label1 = tf.reshape(label1[...,0:4],(-1,4)).numpy();
        label2 = tf.reshape(label2[...,0:4],(-1,4)).numpy();
        label3 = tf.reshape(label3[...,0:4],(-1,4)).numpy();
        label1 = np.concatenate((label1[...,0:2] - label1[...,2:4] / 2,label1[...,0:2] + label1[...,2:4] / 2), axis = -1);
        label2 = np.concatenate((label2[...,0:2] - label2[...,2:4] / 2,label2[...,0:2] + label2[...,2:4] / 2), axis = -1);
        label3 = np.concatenate((label3[...,0:2] - label3[...,2:4] / 2,label3[...,0:2] + label3[...,2:4] / 2), axis = -1);
        img = (image.numpy() * 255.).astype('uint8');
        labels = np.concatenate((label1,label2,label3), axis = 0) * np.array([img.shape[1],img.shape[0],img.shape[1],img.shape[0]]);
        for label in labels:
            cv2.rectangle(img,tuple(label[0:2].astype('int32')),tuple(label[2:4].astype('int32')),(0,255,0),1);
        cv2.imshow('img',img[...,::-1]);
        cv2.waitKey();
        count = count + 1;
        if count == test_num: break;
    print('if you can see the picture and the bounding boxes at the right places, that means the function bbox_to_tensor is OK!');

