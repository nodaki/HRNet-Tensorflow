import glob
import os
from functools import partial

import tensorflow as tf
from omegaconf import DictConfig

AUTOTUNE = tf.data.experimental.AUTOTUNE


def _load_and_preprocess(image_name, label_name):
    image = tf.image.decode_png(tf.io.read_file(image_name))
    label = tf.image.decode_png(tf.io.read_file(label_name))
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    label = tf.cast(label, dtype=tf.int64)
    return image, label


def _resize_image(image, label, is_augment, cfg):
    # Get image shape
    height = tf.cast(tf.shape(image)[0], dtype=tf.float32)
    width = tf.cast(tf.shape(image)[1], dtype=tf.float32)
    # Keep aspect ratio with no padding
    if height > width:
        minval = tf.constant(cfg.TRAINING.IMAGE.WIDTH, dtype=tf.float32) / width
    else:
        minval = tf.constant(cfg.TRAINING.IMAGE.HEIGHT, dtype=tf.float32) / height

    if is_augment:
        # Scaling randomly within 1.0-AUGMENT.RESIZE_MAX_SCALE
        maxval = minval * tf.constant(cfg.TRAINING.AUGMENT.RESIZE_MAX_SCALE, tf.float32)
        scale = tf.random.uniform(shape=(1,), minval=minval, maxval=maxval)
    else:
        scale = minval
    r_height = tf.cast(tf.multiply(height, scale), tf.int32)
    r_width = tf.cast(tf.multiply(width, scale), tf.int32)
    image = tf.image.resize(image, size=tf.concat([r_height, r_width], axis=0), method="bilinear")
    label = tf.image.resize(label, size=tf.concat([r_height, r_width], axis=0), method="nearest")
    return image, label


def _random_crop(image, label, cfg):
    image = tf.image.random_crop(image, size=[cfg.TRAINING.IMAGE.HEIGHT, cfg.TRAINING.IMAGE.WIDTH, 3],
                                 seed=cfg.TRAINING.SEED)
    label = tf.image.random_crop(label, size=[cfg.TRAINING.IMAGE.HEIGHT, cfg.TRAINING.IMAGE.WIDTH, 1],
                                 seed=cfg.TRAINING.SEED)
    return image, label


def augment(image, label, cfg):
    # Random resize
    image, label = _resize_image(image, label, is_augment=True, cfg=cfg)

    # Random crop
    image, label = _random_crop(image, label, cfg=cfg)

    # Random flip
    image = tf.image.random_flip_left_right(image, seed=cfg.TRAINING.SEED)
    label = tf.image.random_flip_left_right(label, seed=cfg.TRAINING.SEED)

    # Random brightness
    image = tf.image.random_brightness(image, max_delta=0.2)
    return image, label


def non_augment(image, label, cfg):
    # Resize
    image, label = _resize_image(image, label, is_augment=False, cfg=cfg)

    # Crop
    image, label = _random_crop(image, label, cfg=cfg)
    return image, label


def create_dataset(cfg: DictConfig, output_filepath, trainval, global_batch_size):
    image_dir = os.path.join(output_filepath, trainval, "image")
    label_dir = os.path.join(output_filepath, trainval, "label")

    # Total image
    num_images = len(glob.glob(image_dir + "/*.png"))

    image_list = tf.data.Dataset.list_files(image_dir + "/*.png", shuffle=False)
    label_list = tf.data.Dataset.list_files(label_dir + "/*.png", shuffle=False)
    image_and_label_list = tf.data.Dataset.zip((image_list, label_list)).shuffle(buffer_size=num_images,
                                                                                 seed=cfg.TRAINING.SEED)
    if trainval == "train2017":
        ds = (
                image_and_label_list
                    .map(_load_and_preprocess, num_parallel_calls=AUTOTUNE)
                    .cache()
                    .map(partial(augment, cfg=cfg), num_parallel_calls=AUTOTUNE)
                    .batch(batch_size=global_batch_size, drop_remainder=True)
                    .prefetch(AUTOTUNE)
        )
    else:
        ds = (
                image_and_label_list
                    .map(_load_and_preprocess, num_parallel_calls=AUTOTUNE)
                    .cache()
                    .map(partial(augment, cfg=cfg), num_parallel_calls=AUTOTUNE)
                    .batch(batch_size=global_batch_size, drop_remainder=True)
                    .prefetch(AUTOTUNE)
        )
    return ds


if __name__ == '__main__':
    pass
