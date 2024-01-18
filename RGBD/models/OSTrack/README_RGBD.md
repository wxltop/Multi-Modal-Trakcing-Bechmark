# This is the README for the [**VOT2022-RGBD challenge**](https://www.votchallenge.net/vot2022/).

## Install The Environment
Use the Anaconda
```
cd OSTrack
conda create -n ostrack python=3.7
conda activate ostrack
bash install.sh
```


## Set Path
Run the following command to set paths for this project
```
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir ./output
```
After running this command, you can also modify paths by editing these two files
```
lib/train/admin/local.py  # paths about training
lib/test/evaluation/local.py  # paths about testing
```


## Download Pretrained Models
Pretrained models can be found here in [model weights](https://drive.google.com/drive/folders/1PwG4i25GZFsB8g5W0E-tZUMUSUlVzcCz?usp=sharing).
Download the pretrained weights under the folder `ostrack320_elimination_cls_t2m12_ep50` and place the weight file under
```$PROJ_ROOT$/output/checkpoints/train/ostrack/ostrack320_elimination_cls_t2m12_ep50```.


## Test and Evaluate OSTrack on The VOT2022
Change the working directory
```
cd external/vot22/OSTrack
```
Modify the **paths** in the [trackers.ini](external/vot22/OSTrack/trackers.ini), then run the experiments with the following command
```
bash exp.sh
```
Please move the automatically downloaded VOT2022 dataset to the **SSD** (solid state disk) for higher IO speeds. 
You can make a soft link for it and put it under `external/vot22/OSTrack` then the code can access it.