echo "=======================Installing basic libraries======================="
sudo apt-get install -y libturbojpeg0
sudo apt-get install -y ninja-build

# install libGL.so.1
sudo apt update
sudo apt install -y libgl1-mesa-glx

# install gcc&g++ 7.4.0
sudo apt-get update
sudo apt-get install -y gcc-7
sudo apt-get install -y g++-7

sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 100
sudo update-alternatives --config gcc

sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-7 100
sudo update-alternatives --config g++
echo "======================================================================="


echo "=========================Installing python libraries========================="
echo "****************** Installing pytorch ******************"
conda install -y pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge

echo "****************** Installing others ******************"
pip install PyYAML easydict cython opencv-python pandas tqdm pycocotools jpeg4py tb-nightly tikzplotlib

pip install --upgrade git+https://github.com/Lyken17/pytorch-OpCounter.git

pip install colorama lmdb scipy visdom

pip install git+https://github.com/votchallenge/vot-toolkit-python

pip install onnx onnxruntime-gpu==1.10.0 timm==0.4.12 yacs einops thop lvis
echo "****************** Installation complete! ******************"

echo "*****************************************************************************"


