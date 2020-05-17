import os

import hydra
import tensorflow as tf
from omegaconf import DictConfig

from src.models.hrnet_segmentation import create_model

# GPU limitation
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


@hydra.main(config_path=os.path.join(os.getenv("PROJECT_DIR"), "src/config/config.yaml"))
def test_create_model(cfg: DictConfig):
    height = cfg.TRAINING.IMAGE.HEIGHT
    width = cfg.TRAINING.IMAGE.WIDTH
    batch_size = cfg.TRAINING.BATCH_SIZE
    model = create_model(cfg)
    inputs = tf.random.uniform((batch_size, height, width, 3))
    outputs = model(inputs)
    # Output shape must be [batch_size, height, width, # of classes]
    assert outputs.get_shape() == tf.TensorShape((batch_size, height, width, cfg.DATASET.NUM_CLASSES))


if __name__ == '__main__':
    test_create_model()
