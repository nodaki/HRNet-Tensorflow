import hydra
import logging
import os
import tensorflow as tf
from omegaconf import DictConfig
from src.features.build_features import create_dataset
from src.models.hrnet_segmentation import create_model
from tensorflow.keras.losses import BinaryCrossentropy, SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.optimizers import Adam

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# GPU limitation
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        logger.info(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        logger.debug(e)


@hydra.main(config_path=os.path.join(os.getenv("PROJECT_DIR", "../"), "src/config/config.yaml"))
def main(cfg: DictConfig):
    # Distributed training
    mirrored_strategy = tf.distribute.MirroredStrategy()
    global_batch_size = int(cfg.TRAINING.BATCH_SIZE * mirrored_strategy.num_replicas_in_sync)

    # Create optimizer
    optimizer = Adam(learning_rate=cfg.TRAINING.LEARNING_RATE.INIT * mirrored_strategy.num_replicas_in_sync)

    # Create loss object
    loss_object = SparseCategoricalCrossentropy(from_logits=True)

    # Create callbacks
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(os.getcwd(), "tensorboard"))
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(os.getcwd(), "model.hdf5"),
            save_best_only=True,
            save_weights_only=True
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(patience=cfg.TRAINING.EARLY_STOPPING)

    # Create dataset
    train_ds = create_dataset(cfg=cfg, output_filepath=os.getenv("OUTPUT_FILEPATH", "../data/processed"),
                              trainval="train2017", global_batch_size=global_batch_size)
    valid_ds = create_dataset(cfg=cfg, output_filepath=os.getenv("OUTPUT_FILEPATH", "../data/processed"),
                              trainval="val2017", global_batch_size=global_batch_size)
    with mirrored_strategy.scope():
        model = create_model(cfg=cfg)
        model.compile(
                optimizer=optimizer,
                loss=loss_object,
                metrics=[SparseCategoricalAccuracy()],
        )
        model.fit(
                x=train_ds,
                epochs=cfg.TRAINING.EPOCHS,
                validation_data=valid_ds,
                callbacks=[tb_callback, early_stopping, checkpoint]
        )


if __name__ == '__main__':
    main()
