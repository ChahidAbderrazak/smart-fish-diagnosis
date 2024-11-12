import os
from argparse import ArgumentParser

from lib import logger
from lib.Autils_Object_detection import deploy_model
from lib.utils import extract_experiment_parameters, load_classes


def prepare_parser():
    parser = ArgumentParser(
        description="Model performance deployment using new non-annotated data"
    )
    parser.add_argument(
        "--list_model",
        default="",
        metavar="FILE",
        help="path to  ensemble_models.json where trained models information",
        type=str,
    )

    parser.add_argument(
        "--image_dir",
        default="",
        metavar="DIRECTORY",
        help="Directory where input images are stored.",
    )

    parser.add_argument(
        "--prediction_dst",
        default="artifacts/deploy_outputs",
        metavar="DIRECTORY",
        help="Directory where input images are stored.",
    )
    parser.add_argument(
        "--cfg",
        default="config/config.yml",
        metavar="FILE",
        help="path to the config.yml file",
        type=str,
    )

    return parser


def get_the_parameters(args):
    # extract the parameters from the config_file
    if not args.cfg == "":
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
        ) = extract_experiment_parameters(config_file=args.cfg)

    if args.image_dir == "":
        image_dir = DIR_DEPLOY
    else:
        image_dir = args.image_dir

    if args.list_model == "":
        model_paths_list = [model_path]
        model_names_list = [model_name]
        ensemble_models_source = f"{args.cfg} [configuration file]"
    else:
        import yaml

        # load list of model form  ensemble_models.json
        with open(args.list_model, "r") as stream:
            ensemble_models = yaml.safe_load(stream)

        # get the list of models and paths
        ensemble_models_source = args.list_model
        model_paths_list = []
        model_names_list = []
        for model_name in ensemble_models.keys():
            model_names_list.append(model_name)
            model_paths_list.append(ensemble_models[model_name])
    # show message
    logger.info(
        f"Loaded {len(model_names_list)} trained model as follows: \
        \n - file source : {ensemble_models_source}\
        \n - model_names_list={model_names_list}\
        \n - model_paths_list={model_paths_list}"
    )
    # load classes
    class_file = os.path.join(os.path.dirname(model_path), "classes.json")
    classes = load_classes(class_file)
    return (
        model_paths_list,
        model_names_list,
        image_dir,
        model_name,
        classes["class_names"],
        size,
    )


def main():
    parser = prepare_parser()
    args = parser.parse_args()

    (
        model_paths_list,
        model_names_list,
        image_dir,
        model_name,
        classes,
        size,
    ) = get_the_parameters(args)

    # deploy the classification model
    pred_images, TS_sz = deploy_model(
        model_paths_list=model_paths_list,
        model_names_list=model_names_list,
        classes=classes,
        size=size,
        DIR_TEST=image_dir,
        DIR_PRED=args.prediction_dst,
        verbose=True,
    )


if __name__ == "__main__":
    main()
