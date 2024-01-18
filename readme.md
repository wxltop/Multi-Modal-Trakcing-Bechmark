# Environment
```cmd
bash install_vipt.sh
conda activate vipt
```
# Dataset
## RGBD:
RGBD主要使用的是 DepthTrack数据集。执行 ./data/download_depthtrack.py 即可下载.

或者去官网下载：[depthtrack](https://github.com/xiaozai/DeT)

## RGBT:
RGBT主要使用 [RGBT234](https://drive.google.com/open?id=1ouNEptXOgRop4U7zYMK9zAp57SZ2XCNL) 和 [LasHeR](https://github.com/BUGPLEASEOUT/LasHeR)。

## RGBE:
RGBE主要使用VisEvent[VisEvent](https://github.com/wangxiao5791509/VisEvent_SOT_Benchmark)

# Methods
## RGBD:
DeT: [DiMP: Learning Discriminative Model Prediction for Tracking](http://openaccess.thecvf.com/content_ICCV_2019/papers/Bhat_Learning_Discriminative_Model_Prediction_for_Tracking_ICCV_2019_paper.pdf)(DeT是DepthTrack论文，项目了用DiMP在DeT中评估)

MixFormer: [MixFormer: End-to-End Tracking with Iterative Mixed Attention](https://arxiv.org/pdf/2203.11082.pdf)

OSTrack: [Joint Feature Learning and Relation Modeling for Tracking: A One-Stream Framework](https://arxiv.org/pdf/2203.11991.pdf)

ProMixTrack 里面还有一层目录：MixFormer

SAMF: A Scale Adaptive Kernel Correlation Filter Tracker with Feature Integration

SPT

## RGBT:
AFPNet: [Attribute-Based Progressive Fusion Network for RGBT Tracking](https://ojs.aaai.org/index.php/AAAI/article/view/20187)

DAFNet: [Deep Adaptive Fusion Network for High Performance RGBT Tracking](https://ieeexplore.ieee.org/document/9021960/)

mfDiMP: [Multi-modal fusion for end-to-end RGB-T tracking](http://openaccess.thecvf.com/content_ICCVW_2019/papers/VOT/Zhang_Multi-Modal_Fusion_for_End-to-End_RGB-T_Tracking_ICCVW_2019_paper.pdf)

MaCNet: [Object Tracking in RGB-T Videos Using Modal-Aware Attention Network and Competitive Learning](https://www.mdpi.com/1424-8220/20/2/393)
## RGBE:
MANet: [Multi-Adapter RGBT Tracking](https://arxiv.org/abs/1907.07485)

MDNet: [RT-MDNet: Real-Time Multi-Domain Convolutional Neural Network Tracker](https://arxiv.org/abs/1808.08834)

VITAL: [VITAL: VIsual Tracking via Adversarial Learning](https://arxiv.org/pdf/1804.04273.pdf)

siamfc: [Fully-Convolutional Siamese Networks for Object Tracking](https://arxiv.org/abs/1606.09549)