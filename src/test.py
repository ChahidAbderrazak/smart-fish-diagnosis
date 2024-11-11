import os
from argparse import ArgumentParser

import numpy as np

from lib import logger
from lib.Autils_Object_detection import load_class_dict, test_model_performance
from lib.networks import get_model_instance
from lib.utils import extract_experiment_parameters


def prepare_parser():
    parser = ArgumentParser(
        description="Model performance deployment  using annotated data"
    )
    parser.add_argument(
        "--cfg",
        default="config/config.yml",
        metavar="FILE",
        help="path to the config.yml file",
        type=str,
    )
    parser.add_argument(
        "--model_path",
        default="",
        metavar="FILE",
        help="path to trained model",
        type=str,
    )

    return parser


def run_model_testing(config_file, model_path="", verbose=False):

    # extract the parameters from the config_file
    (
        device,
        DIR_WORKSPACE,
        RAW_DATA_ROOT,
        dev,
        CSV_TRAIN_FILE,
        CSV_TEST_FILE,
        CSV_DEPLOY_FILE,
        DIR_TRAIN,
        DIR_TEST,
        DIR_DEPLOY,
        size,
        model_name,
        model_path_config,
        num_epoch,
        N_split,
        num_workers,
        step_size,
        gamma,
        transfer_learning,
        momentum,
        weight_decay,
        lr_scheduling,
        verbose,
        lr_list,
        batch_size_list,
        optimizer_list,
        es_patience_ratio_list,
        infer_model_path,
    ) = extract_experiment_parameters(config_file=config_file)

    # replace the model_path_config by the user model_path
    if model_path == "":
        model_path = model_path_config

    #  define the prediction destination folder
    DIR_PRED = os.path.join(os.path.dirname(model_path), "predictions")

    # Test the trained model
    logger.info(
        f"\n\n__________________________________________________________\
            \n#       Testing the model {model_name} using the following "
        + f"parameters: \n#  Device={device} \
            \n#  Dataset={DIR_WORKSPACE}  \
            \n#  Training folder ={DIR_TEST}  \
            \n#  Prediction folder ={DIR_PRED}\
            \n#  Trained model ={model_path}\
            \n__________________________________________________________"
    )
    # Training Loop
    CLASS_dict, classes = load_class_dict(CSV_TEST_FILE)
    # Instantiate the model
    model_arch = get_model_instance(model_name, classes)

    # evaluate the model performance
    model, image_list, test_iou_list = test_model_performance(
        clf_model=model_arch,
        model_path=model_path,
        DIR_TEST=DIR_TEST,
        CSV_TEST_FILE=CSV_TEST_FILE,
        DIR_PRED=DIR_PRED,
        size=size,
        detection_threshold=0.5,
        batch_size=1,
        verbose=verbose,
    )
    iou = np.mean(test_iou_list)
    logger.info(f"\n --> Testing is DONE : \n - avg IOU = {iou}")
    return 0


def main(verbose=True):
    parser = prepare_parser()
    args = parser.parse_args()
    # run the training
    run_model_testing(config_file=args.cfg, model_path=args.model_path)

    # # Testing the model performance using annotated data
    # test_model_performance(
    #     args.model,
    #     args.data,
    #     args.classes_file,
    #     args.annotation,
    #     args.prediction_dst,
    #     verbose=verbose,
    # )


if __name__ == "__main__":
    main()
