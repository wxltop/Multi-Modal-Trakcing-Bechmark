# SPT for VOT challenge
It is the SPT tracker for VOT2022-RGBD challenge.

The code is implemented based on [STARK](https://github.com/researchmm/Stark)



### dependencies
system Ubuntu16.04.6 LTS
GPU RTX3090ti

[cuda11.3](https://developer.download.nvidia.com/compute/cuda/11.3.0/local_installers/cuda_11.3.0_465.19.01_linux.run)

[anaconda3.7](https://www.anaconda.com/distribution/#download-section)

### setup environment
```
conda create -n env_name python=3.6
conda activate env_name
bash install_pytorch17.sh
```
if it raise errors, you can run the command in it one by one    

### download models
download the SPT checkpoint and move it to `/path/to/SPT/lib/train/checkpoints/train/stark_s/rgbd/`

[Google](https://drive.google.com/file/d/1WYBVQ0m-eLJmHVEKFcX2kumu9ZxliHH7/view?usp=sharing)


### setup path

change the `settings.save_dir` and `settings.prj_dir` to corresponding absolute path in the file 
`SPT/test/evaluation/local.py`

change the paths in the file `SPT/tracking/trackers.ini` to corresponding absolute path, and move it to the vot workspace
