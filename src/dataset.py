# structure inspired from with modification and adaptations:
# https://github.com/CSAILVision/semantic-segmentation-pytorch/blob/master/train.py

import os
from argparse import ArgumentParser

import yaml

from lib.utils import search_for_file, split_dataset_train_test


def prepare_parser():
    parser = ArgumentParser(description="Model training")
    parser.add_argument(
        "--cfg",
        default="config/config.yml",
        metavar="FILE",
        help="path to the config.yml file",
        type=str,
    )
    return parser


def run_build_dataset_workspace(config, verbose=False):

    # ------------------------------ Raw Data  -------------------------------
    RAW_DATA_ROOT = config["DATASET"]["raw_dataset"]
    train_split_ratio = config["DATASET"]["train_split_ratio"]
    dev = config["DATASET"]["dev"]
    dev_size = config["DATASET"]["dev_size"]
    DIR_WORKSPACE = config["DATASET"]["workspace"]
    # search for the data.csv file containing the raw data paths + labels
    CSV_DATA_FILE = search_for_file(RAW_DATA_ROOT, file_basename="data.csv")

    # redefine the RAW_DATA_ROOT if the <data.csv> does was not found in
    # the given directory
    RAW_DATA_ROOT = os.path.basename(
        os.path.dirname(os.path.dirname(CSV_DATA_FILE))
    )

    # split dataset train/test
    if config["DATASET"]["new_train_test_split"]:

        split_dataset_train_test(
            CSV_DATA_FILE=CSV_DATA_FILE,
            DIR_WORKSPACE=DIR_WORKSPACE,
            RAW_DATA_ROOT=RAW_DATA_ROOT,
            dev=dev,
            dev_size=dev_size,
            test_size=1 - train_split_ratio,
            ext=".jpg",
            verbose=verbose,
        )
    # # visualize dataset
    # visualize_dataset(DIR_WORKSPACE, verbose=True)


def main():
    parser = prepare_parser()
    args = parser.parse_args()
    # Read config parameters from the sYAML file
    with open(args.cfg, "r") as stream:
        config = yaml.safe_load(stream)
    # Build the workspace
    run_build_dataset_workspace(config)


if __name__ == "__main__":
    main()
