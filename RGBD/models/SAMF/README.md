# Requirements

## Hardware Requirements
* GPU: Tesla V100-SXM2-32GB (32 GB memory)
* CPU: Intel(R) Xeon(R) Platinum 8260 CPU @ 2.40GHz 
* Memory: 376 GB
* HD: 1.7 TB

## Software Dependency
* OS: Ubuntu 18.04 LTS
* Anaconda 3
* CUDA 11.3, CUDNN 7.6.5
* Python=3.9, Pytorch=1.10.1, torchvision=0.11.2 cudatoolkit=11.3.1, and more...
* gcc 7.4.0, g++ 7.4.0

# Install libraries
```Shell
conda create -n mixformer_zhihongfu python=3.9
conda activate mixformer_zhihongfu
bash install.sh
```

# Set up the runtime context.
* Step 1: Download the models from Google Drive (Link: https://drive.google.com/file/d/1eVXiTwprA2Bn00HoBN95_1jbMtyv8KSX/view?usp=sharing), and put the downloaded package `models_SAMixFormer_zhihongfu.zip` in the root directory of our code.


* Step 2: Decompress the downloaded package `models_SAMixFormer_zhihongfu.zip`, rename the decompressed directory `models_SAMixFormer_zhihongfu` to `models`. Once done, the directory structure should be the same as the following:
```Shell
SAMF
├── LICENSE
├── README.md
├── experiments
├── external
├── install.sh
├── lib
├── models
├── rgbd22
└── tracking
```

# Run Evaluation
```Shell
cd rgbd22
vot evaluate --workspace $(pwd) SAMF
```

# Tips
If you get the same error as pictured below, please rerun the evaluation.
![avatar](assets/image1.jpg)

# Contact
If you meet any problem about the environment settings and the code logic, please email <fuzhihong.2022@bytedance.com> ^_^
