import os
from functools import partial

import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE


def _parse_example(example_proto):
    image_feature_description = {
            "image"   : tf.io.FixedLenFeature([], tf.string),
            "label"   : tf.io.FixedLenFeature([], tf.string),
            "height"  : tf.io.FixedLenFeature([], tf.int64),
            "width"   : tf.io.FixedLenFeature([], tf.int64),
            "filename": tf.io.FixedLenFeature([], tf.string),
    }
    return tf.io.parse_single_example(example_proto, image_feature_description)


def _preprocess_for_reshaping(example):
    height = example["height"]
    width = example["width"]
    image = tf.io.decode_raw(example["image"], out_type=tf.float32)
    image = tf.reshape(image, shape=(height, width, 3))
    label = tf.io.decode_raw(example["label"], out_type=tf.int64)
    label = tf.reshape(label, shape=(height, width, 1))
    return image, label


def augment(image, label, cfg):
    # Random crop
    image = tf.image.random_crop(image, size=(cfg.TRAINING.IMAGE.HEIGHT, cfg.TRAINING.IMAGE.WIDTH, 3), seed=1)
    label = tf.image.random_crop(label, size=(cfg.TRAINING.IMAGE.HEIGHT, cfg.TRAINING.IMAGE.WIDTH, 1), seed=1)
    return image, label


def non_augment(image, label, cfg):
    # Random crop
    image = tf.image.random_crop(image, size=(cfg.TRAINING.IMAGE.HEIGHT, cfg.TRAINING.IMAGE.WIDTH, 3), seed=1)
    label = tf.image.random_crop(label, size=(cfg.TRAINING.IMAGE.HEIGHT, cfg.TRAINING.IMAGE.WIDTH, 1), seed=1)
    return image, label


def create_dataset(cfg, output_dir):
    train_path = os.path.join(output_dir, "train2017.tfrecord")
    train_ds = tf.data.TFRecordDataset(train_path)
    train_ds = train_ds.map(_parse_example, num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.map(_preprocess_for_reshaping, num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.map(partial(augment, cfg=cfg), num_parallel_calls=AUTOTUNE).batch(16).prefetch(AUTOTUNE)

    validation_path = os.path.join(output_dir, "val2017.tfrecord")
    val_ds = tf.data.TFRecordDataset(validation_path)
    val_ds = val_ds.map(_parse_example, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.map(_preprocess_for_reshaping, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.map(partial(non_augment, cfg=cfg), num_parallel_calls=AUTOTUNE).batch(16).prefetch(AUTOTUNE)
    return train_ds, val_ds
