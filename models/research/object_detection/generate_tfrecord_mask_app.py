"""
Usage:
  # From tensorflow/models/
  # Create train data:
  python generate_tfrecord.py --csv_input=images/train_labels.csv --image_dir=images/train --output_path=train.record --train_or_test=test

  # Create test data:
  python generate_tfrecord.py --csv_input=images/test_labels.csv  --image_dir=images/test --output_path=test.record  --train_or_test=test
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import pandas as pd
import tensorflow as tf
import numpy as np

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict

# TO-DO replace this with label map
def class_text_to_int(row_label):
    if row_label == 'bay':
        return 1
    else:
        None


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path, mask_path):
    mask_type = 'png'
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size
    error_check = group.filename
    filenameForMask = os.path.splitext(group.filename)[0]
    filenameForMask = os.path.join(mask_path, filenameForMask + '.png')
    mask_img = Image.open(filenameForMask)
    mask_np = np.asarray(mask_img.convert('L'))
    nonbackground_indices_x = np.any(mask_np != 0, axis=0)
    nonbackground_indices_y = np.any(mask_np != 0, axis=1)
    nonzero_x_indices = np.where(nonbackground_indices_x)
    nonzero_y_indices = np.where(nonbackground_indices_y)

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []
    masks = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['label'].encode('utf8'))
        classes.append(class_text_to_int(row['label']))
        mask_remapped = (mask_np != 0).astype(np.uint8)
        masks.append(mask_remapped)
    feature_dict={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }
    if mask_type == 'numerical':
        mask_stack = np.stack(masks).astype(np.float32)
        masks_flattened = np.reshape(mask_stack, [-1])
        feature_dict['image/object/mask'] = (
            dataset_util.float_list_feature(masks_flattened.tolist()))
    elif mask_type == 'png':
        encoded_mask_png_list = []
        for mask in masks:
            img = Image.fromarray(mask)
            output = io.BytesIO()
            img.save(output, format='PNG')
            encoded_mask_png_list.append(output.getvalue())
        feature_dict['image/object/mask'] = (
            dataset_util.bytes_list_feature(encoded_mask_png_list))
    tf_example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    if (int(len(masks))/int(len(classes))) != 1:
        print('ERROR' + error_check)
    return tf_example


def main(image_dir, mask_path, csv_input, output_path):
    writer = tf.python_io.TFRecordWriter(output_path)
    path = image_dir
    mask_path = mask_path
    examples = pd.read_csv(csv_input)
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, path, mask_path)
        writer.write(tf_example.SerializeToString())

    writer.close()
    output_path = output_path
    print('Successfully created the TFRecords: {}'.format(output_path))



