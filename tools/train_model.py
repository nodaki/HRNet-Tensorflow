import os

import hydra
import tensorflow as tf
from omegaconf import DictConfig
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.optimizers import SGD

from features.build_features import create_dataset
from models.hrnet import create_model
from models.metrics import MeanIouWithLogits

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


@hydra.main(config_path=os.path.join(os.getenv("PROJECT_DIR", "../"), "config/config.yaml"))
def main(cfg: DictConfig):
    train_ds, val_ds = create_dataset(cfg=cfg, output_dir=os.getenv("OUTPUT_DIR", "../data/processed"))
    model = create_model(cfg=cfg)

    optimizer = SGD(learning_rate=cfg.TRAINING.LEARNING_RATE, momentum=cfg.TRAINING.MOMENTUM)
    loss_object = SparseCategoricalCrossentropy(from_logits=True)
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=os.getcwd())

    model.compile(
            optimizer=optimizer,
            loss=loss_object,
            metrics=[SparseCategoricalAccuracy(), MeanIouWithLogits(num_classes=cfg.DATASET.NUM_CLASSES)],
    )
    model.fit(train_ds, epochs=cfg.TRAINING.EPOCHS, validation_data=val_ds, callbacks=[tb_callback])


if __name__ == '__main__':
    main()
