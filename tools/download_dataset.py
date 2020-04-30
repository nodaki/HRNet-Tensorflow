import logging
import os
import sys
import time
import zipfile
from tempfile import TemporaryDirectory

import click
import requests


def download_and_unzip(url: str, save_dir: str, overwrite: bool = False):
    """
    Download zipped file and unzip.

    Args:
        url: Download link.
        save_dir: Unzipped files are saved in save_dir.
        overwrite: Overwrite if True
    """
    os.makedirs(save_dir, exist_ok=True)
    name = url.split("/")[-1].split(".")[0]
    filename = os.path.join(save_dir, name)
    if os.path.exists(filename):
        if overwrite:
            logger.info(f"Download {url} and overwrite this")
        else:
            logger.info(f"Skip download {url} because this is already exist")
            return False
    else:
        logger.info(f"Download {url}")

    with TemporaryDirectory() as temp:
        tmp_filename = os.path.join(temp, url.split("/")[-1])
        r = requests.get(url, stream=True)
        total_length = r.headers.get('content-length')
        dl = 0
        start = time.process_time()
        with open(tmp_filename, mode="wb") as f:
            if total_length is None:  # no content length header
                f.write(r.content)
            else:
                for chunk in r.iter_content(1024):
                    dl += len(chunk)
                    f.write(chunk)
                    done = int(50 * dl / float(total_length))
                    sys.stdout.write(
                            "\r[%s%s] %s Mbps" % (
                                    '=' * done, ' ' * (50 - done), dl // (time.process_time() - start) / 10 ** 6))
        logger.info("Download finish")
        with zipfile.ZipFile(tmp_filename) as zf:
            zf.extractall(os.path.join(save_dir, name))


@click.command()
@click.option("--save_dir", "-s", type=str, default=os.getenv("DATA_DIR", "../data/raw"))
def main(save_dir: str):
    """
    Download coco 2017 dataset.

    Args:
        save_dir: Dataset(train, val, annotations) is saved in this directory.
    """
    coco_2017_annotations_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    coco_2017_val_url = "http://images.cocodataset.org/zips/val2017.zip"
    coco_2017_train_url = "http://images.cocodataset.org/zips/train2017.zip"

    download_and_unzip(url=coco_2017_annotations_url, save_dir=save_dir)
    download_and_unzip(url=coco_2017_val_url, save_dir=save_dir)
    download_and_unzip(url=coco_2017_train_url, save_dir=save_dir)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    main()
