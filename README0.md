# Object detection using deep learning models

*  Build dataset from raw data
*  Train the model
*  Deploy the trained model

# Instructions
1. Setup the conda environment
```
$ cd setup-env
$ ./setup_env.sh
```
2. run scripts dataset 
```
python ./scripts/dataset.py --cfg ./config/config.yml
python ./scripts/train.py --cfg ./config/config.yml
python ./scripts/test.py --cfg ./config/config.yml
python ./scripts/deploy.py --cfg ./config/config.yml
```
OR
```
./run_script.sh
```

3. 5un the tensorboard/hyper-parameters 
```
$ tensorboard --logdir <folder path to tensorboard writer>
```
# Acknowledgement

This work was inspired by different online resources. The used parts were adapted with/without modifications. The main resources are cited as follows:
*  [compute IOU](https://gist.github.com/meyerjo/dd3533edc97c81258898f60d8978eddc)
