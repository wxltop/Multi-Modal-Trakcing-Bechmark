
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
sudo apt-get install libturbojpeg
pip install jpeg4py

echo ""
echo ""
echo "****************** Installing tensorboard ******************"
pip install tb-nightly

echo ""
echo ""
echo "****************** Installing tikzplotlib ******************"
pip install tikzplotlib

echo ""
echo ""
echo "****************** Installing colorama ******************"
pip install colorama

echo ""
echo ""
echo "****************** Installing lmdb ******************"
pip install lmdb

echo ""
echo ""
echo "****************** Installing scipy ******************"
pip install scipy

echo ""
echo ""
echo "****************** Installing visdom ******************"
pip install visdom

echo ""
echo ""
echo "****************** Installing vot-toolkit python ******************"
pip install git+https://github.com/votchallenge/vot-toolkit-python


echo ""
echo ""
echo "****************** Installing timm ******************"
pip install timm==0.3.2

echo "****************** Installing yacs/einops/thop ******************"
pip install yacs
pip install einops

echo "****************** Install ninja-build for Precise ROI pooling ******************"
sudo apt-get install ninja-build

echo ""
echo ""
echo "****************** Download model files ep30.pth.tar ******************"
pip install gdown
gdown https://drive.google.com/u/0/uc?id=1tzVlmx2LNEWUlMdJiASlobu3Kgw4mwyO
echo "****************** Download model successfully! ******************"

echo "****************** Installation complete! ******************"