#!/bin/bash
clear
echo && echo " #################################################" 
echo " ##          OBJECT DETECTION PROJECT           " 
echo " ##    config_file=$1"
echo " #################################################" && echo 

if [[ $# -eq 0  ]]; then
    echo && echo "Error: No <config_file> was input as an argument!!!"
    echo "######  Unusual Exit!  ###### " && echo
else
    # #--------------------------------------------------------
    # echo && echo "######  Splitting Dataset for training and evaluation ..."
    # python src/dataset.py --cfg $1

    # #--------------------------------------------------------
    # echo && echo "######  Running model training ..."
    # python src/train.py --cfg $1

    # #--------------------------------------------------------
    # echo && echo "######  Running model evaluation ..."
    # model_path=artifacts/models/OBJ-data_/OBJ_2D_FasterRCNN_ResNet50_imgsz256X256/model_cuda.pth
    # python src/test.py --cfg $1 --model_path $model_path

    #--------------------------------------------------------
    list_model=artifacts/ensemble_models.json
    echo && echo "######  Running model deployment ..."
    python src/deploy.py --cfg $1 --list_model $list_model

fi;


# #### ----------------   NOTIFICATION MESSAGE -------------------------
# notify-send "Execution Finished!!"