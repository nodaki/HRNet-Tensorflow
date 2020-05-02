import collections
import json
import logging
import os

import click
import numpy as np
import tensorflow as tf
from PIL import Image
from omegaconf import OmegaConf, DictConfig
from pycocotools.coco import COCO
from tqdm import tqdm


def load_and_preprocess_image(path: str):
    """Load image and normalize image"""
    image = Image.open(path)
    image = np.array(image, dtype=np.float32)
    image /= 255.0  # normalize to [0,1] range
    return image


def make_tf_example(image: np.ndarray, label: np.ndarray, height: int, width: int, filename: str) -> tf.train.Example:
    return tf.train.Example(features=tf.train.Features(feature={
            "image"   : tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.tostring()])),
            "label"   : tf.train.Feature(bytes_list=tf.train.BytesList(value=[label.tostring()])),
            "height"  : tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
            "width"   : tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
            "filename": tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename.encode()]))
    }))


def create_tf_record_from_coco_annotations(dataset_cfg: DictConfig, data_dir: str, output_dir: str, trainval: str):
    """Create tf record dataset from coco dataset"""
    annotations_file = os.path.join(data_dir, f"annotations_trainval2017/annotations/instances_{trainval}.json")
    coco = COCO(annotation_file=annotations_file)

    if dataset_cfg.DATASET.catNms:
        catNms = dataset_cfg.DATASET.catNms
    else:
        # If not specified categories, all categories are set to target
        catNms = [cat["name"] for cat in coco.loadCats(coco.getCatIds())]
    logger.info(f"Categories: {catNms}")

    catIds = coco.getCatIds(catNms=catNms)
    category_to_label = create_category_to_label(catIds=catIds)
    record_file = os.path.join(output_dir, f"{trainval}.tfrecord")
    recorded_images_count = 0
    with tf.io.TFRecordWriter(record_file) as writer:
        for imgId in tqdm(coco.getImgIds(), desc=f"Make {trainval} tf record"):
            annsIds = coco.getAnnIds(imgIds=imgId, catIds=catIds, iscrowd=None)
            anns = coco.loadAnns(annsIds)
            img = coco.loadImgs(imgId)[0]
            path = os.path.join(data_dir, trainval, trainval, img["file_name"])
            if anns:
                image = load_and_preprocess_image(path)
                # if images is gray scale.
                if image.ndim == 2:
                    image = np.tile(np.expand_dims(image, axis=-1), reps=[1, 1, 3])
                label = np.zeros(shape=(img["height"], img["width"]), dtype=np.int64)
                for ann in anns:
                    label = np.maximum(label, coco.annToMask(ann) * category_to_label[str(ann["category_id"])])
                label = np.expand_dims(label, axis=2)
                tf_example = make_tf_example(image, label, img["height"], img["width"], img["file_name"])
                writer.write(tf_example.SerializeToString())
                recorded_images_count += 1
    logger.info(f"Record counts: {recorded_images_count} @ {trainval}")


def create_category_to_label(catIds):
    """Category to label map"""
    category_to_label = collections.OrderedDict()
    for i, cat_id in enumerate(catIds):
        category_to_label[str(cat_id)] = i + 1
    # Save label map to json file.
    with open(os.path.join(os.getenv("PROJECT_DIR", "../"), "config/category_to_label.json"), "w") as f:
        json.dump(category_to_label, f, indent=4)
    return category_to_label


@click.command()
@click.option("--data_dir", "-d", type=str, default=os.getenv("DATA_DIR", "../data/raw"))
@click.option("--output_dir", "-o", type=str, default=os.getenv("OUTPUT_DIR", "../data/processed"))
@click.option("--dataset_cfg_path", type=str, default="../config/dataset/all_categories.yaml")
def main(data_dir: str, output_dir: str, dataset_cfg_path: str):
    os.makedirs(output_dir, exist_ok=True)
    dataset_cfg = OmegaConf.load(dataset_cfg_path)

    create_tf_record_from_coco_annotations(dataset_cfg, data_dir=data_dir, output_dir=output_dir, trainval="train2017")
    create_tf_record_from_coco_annotations(dataset_cfg, data_dir=data_dir, output_dir=output_dir, trainval="val2017")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    main()
