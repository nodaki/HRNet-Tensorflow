# -*- coding: utf-8 -*-
import click
import collections
import cv2
import json
import logging
import numpy as np
import os
import tensorflow as tf
from omegaconf import OmegaConf, DictConfig
from pathlib import Path
from pycocotools.coco import COCO
from tqdm import tqdm


def create_category_to_label(catNms):
    """Category to label map"""
    category_to_label = collections.OrderedDict()
    for i, cat_nm in enumerate(catNms):
        category_to_label[cat_nm] = i + 1
    # Save label map to json file.
    with open(os.path.join(os.getenv("PROJECT_DIR", "../"), "src/config/category_to_label.json"), "w") as f:
        json.dump(category_to_label, f, indent=4)
    return category_to_label


def make_tf_example(image: np.ndarray, label: np.ndarray, height: int, width: int, filename: str) -> tf.train.Example:
    return tf.train.Example(features=tf.train.Features(feature={
            "image"   : tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.tostring()])),
            "label"   : tf.train.Feature(bytes_list=tf.train.BytesList(value=[label.tostring()])),
            "height"  : tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
            "width"   : tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
            "filename": tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename.encode()]))
    }))


def create_dataset_from_coco_annotations(
        dataset_cfg: DictConfig,
        input_filepath: str,
        output_filepath: str,
        trainval: str,
        tfrecord: bool
):
    """Create tf record dataset from coco dataset"""
    annotations_file = os.path.join(input_filepath, f"annotations_trainval2017/annotations/instances_{trainval}.json")
    coco = COCO(annotation_file=annotations_file)

    # Set target category
    if dataset_cfg.DATASET.catNms:
        catNms = dataset_cfg.DATASET.catNms
    else:
        # If not specified categories, all categories are set to target
        catNms = [cat["name"] for cat in coco.loadCats(coco.getCatIds())]
    logger.info(f"Categories: {catNms}")
    catIds = coco.getCatIds(catNms=catNms)
    category_to_label = create_category_to_label(catNms=catNms)

    # Make image and label dir
    image_dir = Path(os.path.join(output_filepath, trainval, "image"))
    label_dir = Path(os.path.join(output_filepath, trainval, "label"))
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    record_file = os.path.join(output_filepath, f"{trainval}.tfrecord")
    recorded_images_count = 0
    with tf.io.TFRecordWriter(record_file) as writer:
        for imgId in tqdm(coco.getImgIds(), desc=f"Make {trainval} tf record"):
            annsIds = coco.getAnnIds(imgIds=imgId, catIds=catIds, iscrowd=None)
            anns = coco.loadAnns(annsIds)
            img = coco.loadImgs(imgId)[0]
            path = os.path.join(input_filepath, trainval, trainval, img["file_name"])
            if anns:
                image = cv2.imread(path)
                # if images is gray scale.
                if image.ndim == 2:
                    image = np.tile(np.expand_dims(image, axis=-1), reps=[1, 1, 3])
                label = np.zeros(shape=(img["height"], img["width"]), dtype=np.uint8)
                for ann in anns:
                    cat_nm = coco.loadCats(ann["category_id"])[0]["name"]
                    label = np.maximum(label, coco.annToMask(ann) * category_to_label[cat_nm])
                # Save image
                image_file = image_dir / img["file_name"]
                image_file = image_file.with_suffix(".png")
                cv2.imwrite(str(image_file), image)
                label_file = label_dir / img["file_name"]
                label_file = label_file.with_suffix(".png")
                cv2.imwrite(str(label_file), label)
                # Record tfrecord.
                if tfrecord:
                    image = image.astype(np.float32)
                    image /= 255.0  # normalize to [0,1] range
                    label = np.expand_dims(label, axis=2)
                    label = label.astype(np.uint64)
                    tf_example = make_tf_example(image, label, img["height"], img["width"], img["file_name"])
                    writer.write(tf_example.SerializeToString())
                recorded_images_count += 1
    logger.info(f"Record counts: {recorded_images_count} @ {trainval}")


@click.command()
@click.option("--input_filepath", "-d", type=str, default=os.getenv("INPUT_FILEPATH", "../data/raw"))
@click.option("--output_filepath", "-o", type=str, default=os.getenv("OUTPUT_FILEPATH", "../data/processed"))
@click.option("--dataset_cfg_path", type=str, default="../config/dataset/person.yaml")
@click.option("--tfrecord", is_flag=True)
def main(input_filepath: str, output_filepath: str, dataset_cfg_path: str, tfrecord: bool):
    os.makedirs(output_filepath, exist_ok=True)
    dataset_cfg = OmegaConf.load(dataset_cfg_path)

    create_dataset_from_coco_annotations(dataset_cfg, input_filepath=input_filepath, output_filepath=output_filepath,
                                         trainval="train2017", tfrecord=tfrecord)
    create_dataset_from_coco_annotations(dataset_cfg, input_filepath=input_filepath, output_filepath=output_filepath,
                                         trainval="val2017", tfrecord=tfrecord)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    main()
