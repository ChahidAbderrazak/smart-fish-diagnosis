import argparse
import os
import warnings
from argparse import ArgumentParser
from itertools import product

import numpy as np
import pandas as pd

# importing tensorboard
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from lib import logger
from lib.Autils_Object_detection import (
    load_class_dict,
    save_model_for_inference,
    test_model_performance,
    train_model,
)
from lib.networks import Dataset_class, get_model_instance
from lib.utils import (
    collate_fn,
    create_new_folder,
    extract_experiment_parameters,
    get_machine_resources,
    get_time_tag,
    get_train_transform,
    visualize_sample,
)


def prepare_parser():
    parser = ArgumentParser(description="Model training")
    parser.add_argument(
        "--cfg",
        default="config/config.yml",
        metavar="FILE",
        help="path to the config.yml file",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


def save_model_checkpoint(model, model_path):
    """save a checkpoint"""
    import torch

    torch.save(model.state_dict(), model_path)


def save_tensorboard_experiment(
    run_id,
    setup,
    model_arch,
    model_path,
    num_epoch,
    DIR_TRAIN,
    CSV_TRAIN_FILE,
    size,
    verbose,
):
    """
    save Tensorboard experiment setup

    Args:
        run_id (int): experiment run ID
        setup (str): Suffix added to all event filenames
        model_arch (object): loaded model instance architecture
        model_path (path): model path
        num_epoch (int): number of epochs
        DIR_TRAIN (path): training folder
        CSV_TRAIN_FILE (path): path to data.csv
        size (int,int): resize size
        verbose (bool): flag to show the data image sample

    Returns:
        writer: Tensorboard writer
        experiment_dst: destination folder where the experiment outputs
                        will be save
    """
    # save Tensorboard experiment setup
    log_dir = os.path.join(
        os.path.dirname(model_path),
        "runs",
        get_time_tag(type=1)[2:] + setup,
    )
    experiment_TAG = f"epoch{num_epoch}_" + setup
    experiment_dst = os.path.join(
        os.path.dirname(model_path), "runs-pdf", experiment_TAG
    )

    # create the writer
    writer = SummaryWriter(log_dir=log_dir, filename_suffix=setup)

    # find the machine cpu/gpu
    device = get_machine_resources()

    # Model graph/data
    if run_id == 0:
        train_df = pd.read_csv(CSV_TRAIN_FILE)
        train_dataset = Dataset_class(
            dataframe=train_df[:10],
            image_dir=DIR_TRAIN,
            transforms=get_train_transform(size=size),
        )

        # visualize samples
        try:
            if verbose:
                visualize_sample(train_dataset, N=4, verbose=True)
        except Exception as e:
            warnings.warn(f"\n sample cannot be visualized \n {e}")

        try:
            train_loader = DataLoader(
                train_dataset,
                batch_size=4,
                shuffle=False,
                num_workers=0,
                collate_fn=collate_fn,
            )
            images, target, image_ids = next(iter(train_loader))

            # add image grid to tensorboards
            import torchvision

            grid = torchvision.utils.make_grid(images[0])
            writer.add_image("images", grid)
            msg = "\n - image added to the Tensorboard experiment"
            # add the model graph to the Tensorboard
            # TODO: check the graph issue
            images = list(image.to(device) for image in images)
            writer.add_graph(model_arch, list(images[0]))
            msg += "\n - add the model graph to the Tensorboard"
            logger.info(f" -> Tensorboard experiment defined \n{msg}")

        except Exception as e:
            warnings.warn(
                f"\n - cannot visualize a sample of the dataset. \
                    \n This step will be ignored !! \n{e}"
            )
            logger.warn(f"- Some Tensorboard experiment failed except:{msg}")

    return writer, experiment_dst


def run_model_training(config_file, verbose=False):
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
        model_path,
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

    # Hyperparameters dictionary
    parameters = dict(
        lr=lr_list,
        batch_size=batch_size_list,
        optimizer=optimizer_list,
        es_patience_ratio=es_patience_ratio_list,
    )
    param_values = [v for v in parameters.values()]

    # Training Loop
    CLASS_dict, classes = load_class_dict(CSV_TRAIN_FILE)
    for run_id, (lr, batch_size, optimizer, es_patience_ratio) in enumerate(
        product(*param_values)
    ):
        # check the Hyper parameters tuning stage
        if len(list(product(*param_values))) > 1:
            logger.info(
                f" ### Hyper parameters tuning stage [ run id= {run_id}]"
            )

        #  Model Training stage
        logger.info("#####  Model Training Stage #####")

        # model instantiation
        model_arch = get_model_instance(model_name, classes)

        # define the training setup
        setup = (
            f"_bs{batch_size} "
            + f"_lr{lr}optz-{optimizer}"
            + f"_earlystop{es_patience_ratio}"
        )

        # save Tensorboard experiment setup
        writer, experiment_dst = save_tensorboard_experiment(
            run_id=run_id,
            setup=setup,
            model_arch=model_arch,
            model_path=model_path,
            num_epoch=num_epoch,
            DIR_TRAIN=DIR_TRAIN,
            CSV_TRAIN_FILE=CSV_TRAIN_FILE,
            size=size,
            verbose=verbose,
        )

        # train  the model
        model, CLASS_dict, epoch_vect, train_loss_list, val_iou_list = (
            train_model(
                DIR_WORKSPACE=DIR_WORKSPACE,
                RAW_DATA_ROOT=RAW_DATA_ROOT,
                dev=dev,
                model_arch=model_arch,
                model_path=model_path,
                model_name=model_name,
                size=size,
                num_epoch=num_epoch,
                lr=lr,
                batch_size=batch_size,
                momentum=momentum,
                weight_decay=weight_decay,
                save_path=experiment_dst,
                lr_scheduling=lr_scheduling,
                step_size=step_size,
                gamma=gamma,
                optimizer=optimizer,
                es_patience_ratio=es_patience_ratio,
                N_split=N_split,
                num_workers=num_workers,
                transfer_learning=transfer_learning,
            )
        )

        # saved the trained model for inference
        logger.info("--> Saving the trained model for inference")
        save_model_for_inference(
            device=device,
            model_path=infer_model_path,
            model=model,
            model_name=model_name,
            size=size,
            CLASS_dict=CLASS_dict,
            DIR_DEPLOY=DIR_DEPLOY,
            epoch_list=epoch_vect,
            train_loss_list=train_loss_list,
            val_iou_list=val_iou_list,
        )

        # Evaluate the trained model performance
        logger.info("#####  Model Evaluation Stage #####")
        try:
            _, _, test_iou_list = test_model_performance(
                model_arch,
                model_path,
                DIR_TEST,
                CSV_TEST_FILE,
                size=size,
                detection_threshold=0.5,
                batch_size=1,
                verbose=verbose,
            )
        except Exception as e:
            test_iou_list = val_iou_list
            logger.warn(
                f" cannot test trained model performance because of :\
                \n {e}"
            )

        iou = np.mean(test_iou_list)

        # save the checkpoint
        model_path_ACC = os.path.join(
            os.path.dirname(model_path),
            "models-confidence",
            "{:.4f}".format(iou) + "_" + os.path.basename(model_path),
        )
        create_new_folder(os.path.dirname(model_path_ACC))
        save_model_checkpoint(model, model_path_ACC)

        logger.info(
            "--> The model evaluation performance is : \n "
            + f"\n average IOU = {iou}"
        )

        # performance on testing set
        print("________________________________________________________")
        for k in range(len(train_loss_list)):
            writer.add_scalar(
                "Loss/Epochs [training]", train_loss_list[k], epoch_vect[k]
            )
            writer.add_scalar(
                "Loss/Epochs [validation]", val_iou_list[k], epoch_vect[k]
            )
        print("________________________________________________________")

        writer.add_hparams(
            {
                "epochs": len(epoch_vect),
                "lr": lr,
                "bsize": batch_size,
                "optim": optimizer,
                "EarlyStopping": es_patience_ratio,
            },
            {
                "IOU": iou,
                "TestSz": len(test_iou_list),
            },
        )
        print("________________________________________________________")
        del epoch_vect, train_loss_list, val_iou_list

    # close the writer
    writer.close()
    logger.info(
        f"--> The model performance/results is saved in : \
            \n {os.path.dirname(os.path.dirname(experiment_dst))}"
    )

    print("\n --> DONE")
    return model, CLASS_dict


def main():
    parser = prepare_parser()
    args = parser.parse_args()

    # run the training
    run_model_training(config_file=args.cfg)


if __name__ == "__main__":
    main()
