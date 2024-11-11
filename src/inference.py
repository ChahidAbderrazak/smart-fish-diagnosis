import os
import warnings
from argparse import ArgumentParser

import numpy as np
import pandas as pd

# Import PyTorch libraries
import torch

# instantiate the model
import torchvision.transforms as transforms
import yaml
from ensemble_boxes import weighted_boxes_fusion
from matplotlib import pyplot as plt

# Import the model architecture
from lib import logger
from lib.networks import get_model_instance

# define hte device: cpu/cuda
warnings.filterwarnings("ignore")
device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)


# Functions
def getListOfFiles(dirName, ext_list=[".tif", ".tiff", ".jpg"]):
    # create a list of file and sub directories
    # names in the given directory
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
    for path in allFiles:
        ext = "." + path.split(".")[-1]
        if ext not in ext_list:
            allFiles.remove(path)

    return allFiles


def load_classes(class_file):
    import json

    if os.path.exists(class_file):
        with open(class_file) as json_file:
            dict_ = json.load(json_file)
            classes = dict_["class_names"]
    else:
        msg = f"\n The class JSON file ({class_file}) does not exist!!!"
        logger.exception(msg)
        raise Exception(msg)

    return classes


def save_classes(class_file, classes_list):
    try:
        import json

        dict_classes = {k: label for k, label in enumerate(classes_list)}
        dict_ = {}
        dict_["class_names"] = dict_classes
        # create the folder
        create_new_folder(os.path.dirname(class_file))
        # save the classe JSON file

        with open(class_file, "w") as outfile:
            json.dump(dict_, outfile, indent=2)

        return 0

    except Exception as e:
        logger.error(
            f"cannot save classes,json file:\
            \n - class_file={class_file}\
            \n because {e}"
        )
        raise Exception(e)


def create_new_folder(DIR):
    if not os.path.exists(DIR):
        os.makedirs(DIR)


def load_trained_model(clf_model, model_path):
    if os.path.exists(model_path):
        import torch

        device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        clf_model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        msg = f"\n\n - Error: The model path [{model_path}]"
        +" does not exist OR in loading error!!!"
        raise Exception(msg)
    return clf_model


def run_wbf(
    predictions,
    image_index,
    image_size=1024,
    iou_thr=0.001,
    skip_box_thr=0.01,
    weights=None,
):
    boxes = [
        predictions[image_index]["boxes"].data.cpu().numpy() / (image_size - 1)
    ]
    scores = [predictions[image_index]["scores"].data.cpu().numpy()]
    labels = [predictions[image_index]["labels"].data.cpu().numpy()]

    boxes, scores, labels = weighted_boxes_fusion(
        boxes,
        scores,
        labels,
        weights=None,
        iou_thr=iou_thr,
        skip_box_thr=skip_box_thr,
    )
    boxes = boxes * (image_size - 1)
    return boxes, scores, labels


def draw_rectangles(sample, boxes, color=(0, 220, 0), thickness=0):
    import cv2

    # get auto thickness
    if thickness == 0:
        thickness = 1 + np.min(sample.shape) // 200
    for box in boxes:
        cv2.rectangle(
            sample,
            (box[0], box[1]),
            (box[2], box[3]),
            color=color,
            thickness=thickness,
        )
    return sample


def show_detection(img, boxes, title="", save_filename="", plotting=True):
    """plot the detection results the true boxes"""
    img0 = img.copy()
    # draw rectangle on the image
    img_pred_boxes = draw_rectangles(img0, boxes, color=(0, 255, 0))
    # plot the prediction
    fig, axs = plt.subplots(1, 2, figsize=(32, 16))
    axs = axs.ravel()
    # show input image
    axs[0].set_axis_off()
    axs[0].imshow(img)
    axs[0].set_title(f"Input image [ size={img.shape}]")
    # show true target
    axs[0].set_axis_off()
    axs[1].imshow(img_pred_boxes)
    axs[1].set_title(title)
    axs[1].set_axis_off()

    if save_filename != "":
        create_new_folder(os.path.dirname(save_filename))
        plt.savefig(save_filename)

    if plotting:
        plt.show()


def scale_bboxes(boxes, image_w, image_h):
    """
    scale the normalized boxes to the image dimensions

    Args:
        boxes (list): normalized bboxes
        image_w (int): image width
        image_h (int): image height

    Returns:
        boxes: scaled bboxes
    """
    if len(boxes) > 0 and np.max(boxes) <= 1:
        boxes[:, 0], boxes[:, 2] = image_w * boxes[:, 0], image_w * boxes[:, 2]
        boxes[:, 1], boxes[:, 3] = image_h * boxes[:, 1], image_h * boxes[:, 3]

    return boxes.astype(np.int32)


def characterize_detected_objects(boxes, scores, verbose=False):
    area = 0.0
    if len(boxes) > 0:
        for k, box in enumerate(boxes):
            object_width = box[2] - box[0]
            object_height = box[3] - box[1]
            area += object_width * object_height
        area = area / len(boxes)

    #  define the output message
    output_msg = f"average area= {area} pixels"
    if verbose:
        print(f"output_msg={output_msg}")
    return output_msg


def predict_image(
    model_path,
    model_name,
    class_file,
    file_paths,
    DIR_PRED,
    size=(128, 128),
    detection_threshold=0.0,
    plotting=False,
):
    global trained_model
    # get the appropriate device
    if isinstance(file_paths, str):
        file_paths = [file_paths]
    # load classes
    classes = load_classes(class_file)
    # instantiate the model
    clf_model = get_model_instance(model_name, classes=classes)

    # Load the trained model
    trained_model = load_trained_model(clf_model, model_path)
    # Set to eval mode to change behavior of Dropout, BatchNorm
    trained_model.eval()
    # apply transforms
    transform = transforms.Compose([transforms.ToTensor()])

    # run predictions
    nb_detection, pred_scores, outputs = [], [], []
    for file_path in file_paths:
        img = load_resize_convert_image(file_path, size)
        # get the DNN predictions
        output, boxes, scores = get_detection_boxes(
            img, size, transform, detection_threshold=detection_threshold
        )
        # append the results:
        outputs.append(output)  # Saving outputs and scores
        nb_detection.append(len(boxes))
        pred_scores.append(np.mean(np.array(scores)))
        # characterize the road damages:
        output_msg = characterize_detected_objects(boxes, scores)
        # plot the detection results the true boxes
        save_filename = os.path.join(
            DIR_PRED,
            f"obj-detected{len(boxes)}_" + os.path.basename(file_path),
        )
        title = f"{len(boxes)} Detection: {output_msg}"
        show_detection(
            (img * 255).astype(np.uint8),
            boxes,
            title=title,
            save_filename=save_filename,
            plotting=plotting,
        )

    # create output pandas
    prediction_df = create_dataframe(file_paths, nb_detection, pred_scores)
    return prediction_df


def get_detection_boxes(img, size, transform, detection_threshold=0.0):
    x = transform(img)  # Preprocess image
    x = x.unsqueeze(0)  # Add batch dimension
    # get predictions
    if device == torch.device("cuda"):
        x = x.cuda()
        # trained_model(data).cpu().argmax(1):
        predictions = trained_model(x).cpu()  # Forward pass
    else:
        predictions = trained_model(x)  # Forward pass
    # get the predicted boxes
    boxes, scores, labels = run_wbf(
        predictions, image_index=0, image_size=size[0]
    )
    # input(f'\n - boxes={boxes}')
    # boxes = boxes.astype(np.int32).clip(min=0, max=size[0] - 1)
    preds_sorted_idx = np.argsort(scores)[::-1]
    preds_sorted = boxes[preds_sorted_idx]
    # apply score threshold
    boxes = boxes[scores >= detection_threshold].astype(np.int32)
    # save the predicted boxes and scores
    output = {"boxes": preds_sorted, "scores": scores}

    return output, boxes, scores


def predict_video(
    model_path,
    model_name,
    class_file,
    video_path,
    DIR_PRED,
    size=(128, 128),
    detection_threshold=0.5,
    plotting=False,
):
    global trained_model
    # load classes
    classes = load_classes(class_file)
    # instantiate the model
    clf_model = get_model_instance(model_name, classes=classes)
    # Load the trained model
    trained_model = load_trained_model(clf_model, model_path)
    # Set to eval mode to change behavior of Dropout, BatchNorm
    trained_model.eval()
    # apply transforms
    transform = transforms.Compose([transforms.ToTensor()])

    # detect objects in  each frame  and save the results
    predict_video_frames(
        video_path=video_path,
        transform=transform,
        size=size,
        DIR_PRED=DIR_PRED,
        plotting=plotting,
    )

    return 0


def predict_video_frames(video_path, transform, size, DIR_PRED, plotting):
    """
    detect objects in  each frame  and save the results

    Args:
        video_path (path): video path
        transform (torch transform): needed image transform
        size (truple): resizing image size
        DIR_PRED (path): destination folder where predictions will be saved
        plotting (bool): show the detection while running

    Returns:
        _type_: _description_
    """
    # load the video file
    import cv2

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    # save the output video file
    video_filename = os.path.basename(video_path)
    extension = os.path.splitext(video_filename)[1]
    video_path_out = os.path.join(
        DIR_PRED, video_filename.replace(extension, "_out.avi")
    )
    out = cv2.VideoWriter(
        video_path_out, cv2.VideoWriter_fourcc(*"XVID"), fps, size
    )

    # Check if camera opened successfully
    if not cap.isOpened():
        msg = f"Error opening video file: {video_path}"
        logger.exception(msg)
        raise Exception(msg)
    else:
        logger.info(
            f"video file loaded correctly [fps={fps}]:\
              \n - file={video_path} "
        )

    # run predictions
    nb_detection, pred_scores, outputs = [], [], []
    show_warnings = True
    skipped_frame = 0
    frame_cnt = 0

    # Read until video is completed
    while cap.isOpened():
        try:
            # Capture frame-by-frame
            ret, frame0 = cap.read()
            frame_cnt += 1
            if len(np.unique(frame0)) == 1:

                if skipped_frame == 0:
                    logger.info(" - frame skipped because it is empty!!")
                skipped_frame += 1
                continue

            #  resize the loaded frame
            frame = resize_convert_video_frame(frame0, size)

            # get the model predictions
            output, boxes, scores = get_detection_boxes(
                frame, size, transform, detection_threshold=0.0
            )
            boxes_normalized = boxes / size[0]
            # append the results:
            outputs.append(output)  # Saving outputs and scores
            nb_detection.append(len(boxes))
            pred_scores.append(np.mean(np.array(scores)))
            # print(f"number of detected object ={len(boxes)}")

            # restore [0, 255] value range of the  resized frame
            frame = (255 * frame).astype(np.uint16)
            # add detected object boxes
            if len(boxes) > 0:
                frame_boxes = draw_rectangles(frame, boxes, color=(0, 255, 0))

                # scale the normalized boxes to the image dimensions
                image_h, image_w, channels = frame0.shape
                scaled_boxes = scale_bboxes(boxes_normalized, image_w, image_h)

                frame_boxes = draw_rectangles(
                    frame0, scaled_boxes, color=(0, 255, 0)
                )

                # characterize the detected objects:
                output_msg = characterize_detected_objects(boxes, scores)
                print(f"output_msg={output_msg}")

            else:
                # save the original frame without predictions in the video file
                frame_boxes = frame0

            # save the frame with prediction in the output video file
            out.write(frame_boxes)
            cv2.imwrite(video_path_out + ".jpg", frame_boxes)

            # Display the resulting frame
            try:
                if plotting and ret:
                    # Display the prediction frame with bboxes
                    cv2.imshow("Frame", frame_boxes)

                    # Press Q on keyboard to exit
                    if cv2.waitKey(25) & 0xFF == ord("q"):
                        break  # Break the loop
                else:
                    continue

            except Exception as e:
                if show_warnings:
                    logger.warn(
                        "prediction video cannot be showed on stream!!"
                        + f" because \n {e}"
                    )
                    show_warnings = False
                continue
        except KeyboardInterrupt:  # break the loop when "Ctrl-C is pressed!")
            break

    # Quit the  viewer
    cap.release()
    cv2.destroyAllWindows()

    logger.info(
        f"--> prediction video finished: \
        \n - skipped_frame={skipped_frame}"
        + f" [{round(100*skipped_frame/frame_cnt, 2)}%]\
            \n output video={video_path_out}"
    )

    return 0


def load_resize_convert_image(file_path, size):
    import cv2

    image = cv2.imread(file_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    image /= 255.0
    # resize image
    img = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    return img


def resize_convert_video_frame(frame, size):
    import cv2

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)
    image /= 255.0
    # resize image
    img = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    return img


def plot_image(img, title, filename=""):
    # # import Image
    import matplotlib.pyplot as plt

    # for img, filename in zip(imgs, filenames):
    plt.axis("off")
    plt.imshow(img)
    plt.title(title + "\nfile = " + filename)
    plt.show()


def create_dataframe(file_paths, nb_detection, pred_score):
    # print(
    #     f"\n - files={len(file_paths)}\
    #     \n - nb_detection={len(nb_detection)}\
    #     \n - pred_score={len(pred_score)}\n "
    # )
    dict = {
        "file": file_paths,
        "nb-detections": nb_detection,
        "Confidence-percentage": pred_score,
    }

    return pd.DataFrame(dict)


def prepare_parser():
    parser = ArgumentParser(description="Model deployment")
    parser.add_argument(
        "--model_path",
        required=True,
        metavar="FILE",
        help="path to the trained model <.pth>",
        type=str,
    )
    parser.add_argument(
        "--info",
        default="model/info.json",
        metavar="FILE",
        help="path to the config.yml file",
        type=str,
    )

    parser.add_argument(
        "--data_source",
        required=True,
        metavar="DIRECTORY",
        help="Directory where input images are stored.",
    )
    parser.add_argument(
        "--prediction_dst",
        default="predictions",
        metavar="DIRECTORY",
        help="Directory where results will be stored.",
    )
    parser.add_argument(
        "--plot", default="1", help="show the image with its prediction."
    )

    return parser


def get_the_parameters(args):

    # list the image to be predicted
    ext_list = [".tif", ".tiff", ".jpg"]

    # Read config parameters from the sYAML file
    with open(args.info, "r") as stream:
        model_info = yaml.safe_load(stream)

    model_path = args.model_path
    model_name = model_info["model_name"]
    size = (model_info["size"][0], model_info["size"][1])
    class_file = os.path.join(os.path.dirname(model_path), "classes.json")
    prediction_dst = args.prediction_dst

    # get the potting flag
    if args.plot == "0":
        plotting = False
    else:
        plotting = True

    return (
        model_path,
        model_name,
        size,
        class_file,
        prediction_dst,
        ext_list,
        plotting,
    )


def model_inference_using_images(args):
    # extract the parameters
    (
        model_path,
        model_name,
        size,
        class_file,
        prediction_dst,
        ext_list,
        plotting,
    ) = get_the_parameters(args)

    detection_threshold = 0.0
    # display
    logger.info(
        f"\n______________________________________________________\
        \n#                      MODEL  INFERENCE\
        \n# model_name ={model_name}\
        \n#  Trained model ={model_path}\
        \n#  Data folder={args.data_source} \
        \n#  Prediction folder ={prediction_dst} \
         \n______________________________________________________"
    )

    # find images
    list_images_paths = getListOfFiles(args.data_source, ext_list)
    logger.info(f"{len(list_images_paths)} images were found")

    # deploy the classification model
    prediction_df = predict_image(
        model_path,
        model_name,
        class_file,
        list_images_paths[0:],
        DIR_PRED=prediction_dst,
        size=size,
        detection_threshold=detection_threshold,
        plotting=plotting,
    )
    logger.info(f"prediction results : \n {prediction_df}")

    # save result tables
    filename_csv = os.path.join(prediction_dst, "summary.csv")
    create_new_folder(os.path.dirname(filename_csv))
    prediction_df.to_csv(filename_csv, sep=",")

    logger.info(
        f"\nprediction result table is saved in : \
        \n - Predicted images : {prediction_dst}\
        \n - summary.csv : {filename_csv}\n "
    )

    return 0


def model_inference_using_video(args):
    # extract the parameters
    (
        model_path,
        model_name,
        size,
        class_file,
        prediction_dst,
        ext_list,
        plotting,
    ) = get_the_parameters(args)

    detection_threshold = 0.0

    detection_threshold = 0.5

    # display
    logger.info(
        f"\n________________________________________________________\
        \n#                      MODEL  INFERENCE\
        \nname ={model_name}\
        \n#  Trained model ={model_path}\
        \n#  Data folder={args.data_source} \
        \n#  Prediction folder ={prediction_dst} \
        \n________________________________________________________n\n"
    )

    # find images
    video_path = args.data_source
    logger.info(f" The input video= {video_path} ")

    # deploy the classification model
    predict_video(
        model_path,
        model_name,
        class_file,
        video_path,
        DIR_PRED=prediction_dst,
        size=size,
        detection_threshold=detection_threshold,
        plotting=plotting,
    )

    return 0


if __name__ == "__main__":
    parser = prepare_parser()
    args = parser.parse_args()

    #  choose whether t infer a video or an image
    if os.path.isdir(args.data_source):
        # run images inference
        model_inference_using_images(args)

    elif os.path.isfile(args.data_source):
        # run video inference
        model_inference_using_video(args)

    else:
        logger.error(
            " the data_source argument is neither"
            + f" a video nor an image folder! \
                \n - data_source = {args.data_source} \
                \n\n -> please recheck <--data_source> availability"
            + " and assign it correctly!"
        )
