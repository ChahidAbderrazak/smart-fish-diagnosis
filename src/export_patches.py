import argparse
import os
from argparse import ArgumentParser
from glob import glob

import numpy as np
import pandas as pd
import yaml
from matplotlib import pyplot as plt

from lib import logger
from lib.networks import Dataset_class
from lib.utils import get_directories, get_valid_transform


def prepare_parser():
    parser = ArgumentParser(description="Export object patches")
    parser.add_argument(
        "--cfg",
        default="config/config.yml",
        metavar="FILE",
        help="path to the config.yml file",
        type=str,
    )
    parser.add_argument(
        "--dst",
        metavar="DIR",
        required=True,
        help="path to patches destination folder",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


def search_class_dict(ROOT):
    import json

    class_file = glob(os.path.join(ROOT, "classes.json"))
    for k in range(3):
        sub_fld = ["*" for i in range(k + 1)]
        class_file += glob(
            os.path.join(ROOT, "/".join(sub_fld), "classes.json")
        )
    if len(class_file) == 0:
        return {}
    else:
        with open(class_file[0]) as f:
            CLASS_dict = json.load(f)
        return CLASS_dict


def save_patches_from_labeled_dataset(
    dst, dataset_obj, CLASS_dict, verbose=True
):
    """
    visualize image in grid <NxN>
    """
    import cv2

    # prepare images indexes
    list_idx = [i for i in range(len(dataset_obj))]

    # Show selected images
    for k, idx in enumerate(list_idx):
        image, target, image_id = dataset_obj.__getitem__(idx)
        img = image.permute(1, 2, 0).cpu().numpy()
        boxes = target["boxes"].cpu().numpy().astype(np.int32)
        boxes_label = target["labels"].cpu().numpy()
        color = (255, 0, 0)
        line_thickness = int(np.max(image.shape) / 200)
        if line_thickness == line_thickness:
            line_thickness = 1
        # plot the boxes
        for i, box in enumerate(boxes):
            xmin, xmax = box[0], box[2]
            ymin, ymax = box[1], box[3]
            patch = 255 * img[ymin:ymax, xmin:xmax, :]
            patch_pixels_value = np.unique(patch)
            # print(
            #     f"\n box={box} patch_pixels_value:\
            #       \n {patch_pixels_value} \n box={box}"
            # )

            # save the patch image
            if len(patch_pixels_value) > 1:
                idx_lbl = boxes_label[i]
                try:
                    label_ = CLASS_dict[f'"{str(idx_lbl)}"']
                except Exception:
                    label_ = CLASS_dict[f"{str(idx_lbl)}"]
                filename = os.path.join(dst, f"{label_}_-_{image_id}_{i}.png")
                try:
                    cv2.imwrite(filename, patch)
                    logger.info(f" --> New patches saved in :\n {filename}")
                except Exception:
                    msg = f"Cannot save [{filename}]. \
                            \n The format {filename[-4:]} is unsupported!!! "
                    logger.error(msg)
                    raise UserWarning(msg)
            # draw boxes
            cv2.rectangle(
                img,
                (box[0], box[1]),
                (box[2], box[3]),
                color,
                thickness=line_thickness,
            )

        if verbose:
            plt.figure()
            tag_image = (
                "image ["
                + str(image_id)
                + "] , size ["
                + str(image.shape)
                + "]"
            )
            plt.title(str(tag_image))
            plt.imshow(img)
            plt.show()


def export_patchs(config, dst, verbose=False):
    WORKSPACE_folder = config["DATASET"]["workspace"]
    dev = config["DATASET"]["dev"]
    RAW_DATA_ROOT = config["DATASET"]["raw_dataset"]
    # get/load the files paths and directories
    (
        DIR_WORKSPACE,
        CSV_TRAIN_FILE,
        CSV_TEST_FILE,
        CSV_DEPLOY_FILE,
        DIR_TRAIN,
        DIR_TEST,
        DIR_DEPLOY,
        data_TAG,
    ) = get_directories(WORKSPACE_folder, RAW_DATA_ROOT, dev=dev)
    train_df = pd.read_csv(CSV_TRAIN_FILE)
    CLASS_dict = search_class_dict(ROOT=DIR_TRAIN)

    # training dataLoading
    train_dataset = Dataset_class(
        train_df[:10],
        DIR_TRAIN,
        get_valid_transform(size=(-1, -1)),
        CLASS_dict=CLASS_dict,
    )
    # visualize samples
    save_patches_from_labeled_dataset(
        dst=dst,
        dataset_obj=train_dataset,
        CLASS_dict=CLASS_dict,
        verbose=verbose,
    )
    return 0


def main():
    parser = prepare_parser()
    args = parser.parse_args()
    # Read config parameters from the sYAML file
    with open(args.cfg, "r") as stream:
        config = yaml.safe_load(stream)
    # run the training
    export_patchs(config, args.dst, verbose=False)


if __name__ == "__main__":
    main()
