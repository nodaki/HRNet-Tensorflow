import tensorflow as tf


class MeanIouWithLogits(tf.keras.metrics.MeanIoU):
    """Mean IoU with logits
    https://github.com/tensorflow/tensorflow/issues/32875#issuecomment-542932089
    """
    __name__ = "MeanIoU"

    def __call__(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        return super().__call__(y_true, y_pred, sample_weight=sample_weight)
