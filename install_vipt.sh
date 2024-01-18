# conda 安装太慢：https://blog.csdn.net/qazplm12_3/article/details/108924561
echo "***************** create environment *******************"
conda create -n vipt python=3.8
conda activate vipt

echo "****************** Installing pytorch ******************"
# conda install -y pytorch==1.7.0 torchvision==0.8.1 cudatoolkit=10.2 -c pytorch
conda install -y pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch

echo ""
echo ""
echo "****************** Installing yaml ******************"
pip install PyYAML

echo ""
echo ""
echo "****************** Installing easydict ******************"
pip install easydict

echo ""
echo ""
echo "****************** Installing cython ******************"
pip install cython

echo ""
echo ""
echo "****************** Installing opencv-python ******************"
pip install opencv-python

echo ""
echo ""
echo "****************** Installing pandas ******************"
pip install pandas

echo ""
echo ""
echo "****************** Installing tqdm ******************"
pip install tqdm

echo ""
echo ""
echo "****************** Installing coco toolkit ******************"
pip install pycocotools

echo ""
echo ""
echo "****************** Installing jpeg4py python wrapper ******************"
apt-get install libturbojpeg
pip install jpeg4py

echo ""
echo ""
echo "****************** Installing scipy ******************"
pip install scipy

echo ""
echo ""
echo "****************** Installing timm ******************"
pip install timm==0.5.4

echo ""
echo ""
echo "****************** Installing tensorboard ******************"
pip install tb-nightly

echo ""
echo ""
echo "****************** Installing lmdb ******************"
pip install lmdb

echo ""
echo ""
echo "****************** Installing visdom ******************"
pip install visdom

echo ""
echo ""
echo "****************** Installing vot-toolkit python ******************"
# pip install git+https://github.com/votchallenge/vot-toolkit-python
pip install vot-toolkit==0.5.3
pip install vot-trax==3.0.3

echo ""
echo "****************** Installing other pakages *********************"
pip install jpeg4py
pip install lmdb
pip install pandas
pip install pycocotools
pip install lvis
pip install wget
pip install shapely
pip install scikit-learn
pip install einops
pip install yacs

echo "****************** Installation complete! ******************"