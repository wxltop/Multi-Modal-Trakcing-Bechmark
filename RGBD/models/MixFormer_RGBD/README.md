# MixFormer-RGBD
The MixFormer-RGBD tracker for the VOT2022-RGBD challenge

## Install the environment
Environment Requirements:
- Operating System: Ubuntu 18.04
- CUDA: 11.1 and 10.2 are tested
- GCC: 8.4.0 and 7.5.0 are tested
- Dependencies: All dependencies are specified in `install.sh`
- Device: Intel® Core™ i9-9900K CPU @ 3.60GHz × 16 and *GeForce RTX 2080 Ti* 

**Environment Setup**: Use the Anaconda

```
# create environment
conda create -n mixformer python=3.7
conda activate mixformer

# install pytorch with conda (not recommend)
conda install -y pytorch==1.7.0 torchvision==0.8.1 cudatoolkit=10.2 -c pytorch
# install pytorch whl that is already complied(recommend)
## go to https://download.pytorch.org/whl/torch_stable.html and download whls are compatible with your cuda version and install them by:
pip install torch-1.7.0-cp37-cp37m-linux_x86_64.whl
pip install torchvision-0.8.1-cp37-cp37m-linux_x86_64.whl

# install dependancies and download model files
bash install.sh

# set up paths for the project
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir .
```

## Test and evaluate MixFormer-RGBD on vot2022/rgbd sequences

**VOT2022-RGBD**

- Download vot2022/rgbd sequences and put them under `<PROJECT_DIR>/external/vot2022rgbd/mixformer_large`
- Modify the `env_PATH` to yours on your local machine in [trackers.ini](external/vot2022rgbd/mixformer_large/trackers.ini)

- Then run the following script.
```
cd external/vot2022rgbd/mixformer_large
bash set_prj_path.sh

vot evaluate --workspace . mixformerrgbd_large_rgbd
vot analysis --workspace . mixformerrgbd_large_rgbd  --nocache --format html
vot pack --workspace . mixformerrgbd_large_rgbd  # for submission
```
Note: The <PROJECT_DIR> in [trackers.ini](external/vot2022rgbd/mixformer_large/trackers.ini) requires absolute path  of the MixFormer-RGBD project on your local machine. Our [run.sh](external/vot2022depth/mixformer_large/run.sh) can automatically locate the <PROJECT_DIR>. However, you can also manually specify <PROJECT_DIR> in [trackers.ini](external/vot2022rgbd/mixformer_large/trackers.ini)



## Runtime Tips (Important)
When the PrRoIPooling module *is compiled for the first time*, the following exception may be raised.

    `external/PreciseRoIPooling/pytorch/prroi_pool/src/prroi_pooling_gpu.c: In function`

The solution is **re-run the evaluation command again**.
