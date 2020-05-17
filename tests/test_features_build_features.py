import os

import tensorflow as tf
from omegaconf import OmegaConf

from src.features.build_features import create_dataset


def _check_shape(ds, image_shape, label_shape):
    image, label = next(iter(ds))
    assert image.get_shape() == tf.TensorShape(image_shape)
    assert label.get_shape() == tf.TensorShape(label_shape)


def test_create_dataset():
    cfg = OmegaConf.load(os.path.join(os.getenv("PROJECT_DIR"), "src/config/training/hrnet_w32_segmentation.yaml"))
    batch_size = cfg.TRAINING.BATCH_SIZE
    height = cfg.TRAINING.IMAGE.HEIGHT
    width = cfg.TRAINING.IMAGE.WIDTH
    image_shape = [batch_size, height, width, 3]
    label_shape = [batch_size, height, width, 1]
    train_ds = create_dataset(cfg, output_filepath=os.getenv("OUTPUT_FILEPATH"), trainval="train2017",
                              global_batch_size=batch_size)
    valid_ds = create_dataset(cfg, output_filepath=os.getenv("OUTPUT_FILEPATH"), trainval="val2017",
                              global_batch_size=batch_size)
    _check_shape(train_ds, image_shape, label_shape)
    _check_shape(valid_ds, image_shape, label_shape)
