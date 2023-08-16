# Spatial Transform Decoupling for Oriented Object Detection

## Abstract

<div align=center><img src="./figures/framework.png"></div>

Vision Transformers (ViTs) have achieved remarkable success in computer vision tasks. However, their potential in rotation-sensitive scenarios has not been fully explored, and this limitation may be inherently attributed to the lack of spatial invariance in the data-forwarding process. In this study, we present a novel approach, termed Spatial Transform Decoupling (STD), providing a simple-yet-effective solution for oriented object detection with ViTs. Built upon stacked ViT blocks, STD utilizes separate network branches to predict the position, size, and angle of bounding boxes, effectively harnessing the spatial transform potential of ViTs in a divide-and-conquer fashion. Moreover, by aggregating cascaded activation masks (CAMs) computed upon the regressed parameters, STD gradually enhances features within regions of interest (RoIs), which complements the self-attention mechanism. Without bells and whistles, STD achieves state-of-the-art performance on the benchmark datasets including DOTA-v1.0 (82.24\% mAP) and HRSC2016 (98.55\% mAP), which demonstrates the effectiveness of the proposed method. Source code is enclosed in the supplementary material. Source code is available at https://github.com/yuhongtian17/Spatial-Transform-Decoupling.

Full paper is available at https://arxiv.org/pdf/2308.xxxxx.pdf.

## Results and models

Imagenet MAE pre-trained ViT-S backbone: [pan.baidu.com](https://pan.baidu.com/s/19nw-Ry2pGoeHZ0lQ-XehQg)

Imagenet MAE pre-trained ViT-B backbone: [official MAE weight](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base_full.pth)

Imagenet MAE pre-trained HiViT-B backbone: [pan.baidu.com](https://pan.baidu.com/s/19nw-Ry2pGoeHZ0lQ-XehQg)

> Password of `pan.baidu.com`: STDC

DOTA-v1.0 (multi-scale)

|               Model                |  mAP  | Angle | lr schd | Batch Size | Configs | Models |  Logs  | Submissions |
| :--------------------------------: | :---: | :---: | :-----: | :--------: | :-----: | :----: | :----: | :---------: |
|  STD with Oriented RCNN and ViT-B  | 81.66 | le90  |   1x    |    1\*8    | [cfg](./mmrotate-main/configs/rotated_imted/dota/vit/rotated_imted_vb1m_oriented_rcnn_vit_base_1x_dota_ms_rr_le90_stdc_xyawh321v.py) | [model](https://pan.baidu.com/s/19nw-Ry2pGoeHZ0lQ-XehQg) | [log](https://pan.baidu.com/s/19nw-Ry2pGoeHZ0lQ-XehQg) | [submission](https://pan.baidu.com/s/19nw-Ry2pGoeHZ0lQ-XehQg) |
| STD with Oriented RCNN and HiViT-B | 82.24 | le90  |   1x    |    1\*8    | [cfg](./mmrotate-main/configs/rotated_imted/dota/hivit/rotated_imted_hb1m_oriented_rcnn_hivitdet_base_1x_dota_ms_rr_le90_stdc_xyawh321v.py) | [model](https://pan.baidu.com/s/19nw-Ry2pGoeHZ0lQ-XehQg) | [log](https://pan.baidu.com/s/19nw-Ry2pGoeHZ0lQ-XehQg) | [submission](https://pan.baidu.com/s/19nw-Ry2pGoeHZ0lQ-XehQg) |

HRSC2016

|               Model                | mAP(07) | mAP(12) | Angle | lr schd | Batch Size | Configs | Models |  Logs  |
| :--------------------------------: | :-----: | :-----: | :---: | :-----: | :--------: | :-----: | :----: | :----: |
|  STD with Oriented RCNN and ViT-B  |  90.67  |  98.55  | le90  |   3x    |    1\*8    | [cfg](./mmrotate-main/configs/rotated_imted/hrsc/vit/rotated_imted_oriented_rcnn_vit_base_3x_hrsc_rr_le90_stdc_xyawh321v.py) | [model](https://pan.baidu.com/s/19nw-Ry2pGoeHZ0lQ-XehQg) | [log](https://pan.baidu.com/s/19nw-Ry2pGoeHZ0lQ-XehQg) |
| STD with Oriented RCNN and HiViT-B |  90.63  |  98.20  | le90  |   3x    |    1\*8    | [cfg](./mmrotate-main/configs/rotated_imted/hrsc/hivit/rotated_imted_oriented_rcnn_hivitdet_base_3x_hrsc_rr_le90_stdc_xyawh321v.py) | [model](https://pan.baidu.com/s/19nw-Ry2pGoeHZ0lQ-XehQg) | [log](https://pan.baidu.com/s/19nw-Ry2pGoeHZ0lQ-XehQg) |

## Installation

[MMRotate](https://github.com/open-mmlab/mmrotate) depends on [PyTorch](https://pytorch.org/), [MMCV](https://github.com/open-mmlab/mmcv) and [MMDetection](https://github.com/open-mmlab/mmdetection).
Please refer to [Install Guide](https://mmrotate.readthedocs.io/en/latest/install.html) for more detailed instruction.
Below are quick steps for installation.

```shell
conda create -n openmmlab python=3.7 -y
conda activate openmmlab
conda install pytorch=1.7.0 torchvision torchaudio cudatoolkit=10.2 -c pytorch
pip install openmim
mim install mmcv-full==1.6.1
mim install mmdet==2.25.1
git clone https://github.com/open-mmlab/mmrotate.git
cd mmrotate
pip install -r requirements/build.txt
pip install -v -e .
cd ../
# 
# pip install timm apex
# 
git clone https://github.com/yuhongtian17/Spatial-Transform-Decoupling.git
cp -r Spatial-Transform-Decoupling/mmrotate-main/* mmrotate/
```

If you want to conduct offline testing on the DOTA-v1.0 dataset (for example, our ablation study is trained on the train-set and tested on the val-set), we recommend using the official [DOTA devkit](https://github.com/CAPTAIN-WHU/DOTA_devkit). Here we modify the evaluation code for ease of use.

```shell
git clone https://github.com/CAPTAIN-WHU/DOTA_devkit.git
cd DOTA_devkit
sudo apt install swig
swig -c++ -python polyiou.i
python setup.py build_ext --inplace
cd ../
# 
git clone https://github.com/yuhongtian17/Spatial-Transform-Decoupling.git
cp Spatial-Transform-Decoupling/DOTA_devkit-master/dota_evaluation_task1.py DOTA_devkit/
```

Example usage:

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./tools/dist_train.sh ./configs/rotated_faster_rcnn/rotated_faster_rcnn_r50_fpn_1x_dota_le90.py 8
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup ./tools/dist_train.sh ./configs/rotated_faster_rcnn/rotated_faster_rcnn_r50_fpn_1x_dota_le90.py 8 > nohup.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./tools/dist_test.sh ./configs/rotated_faster_rcnn/rotated_faster_rcnn_r50_fpn_1x_dota_le90.py ./work_dirs/rotated_faster_rcnn_r50_fpn_1x_dota_le90/epoch_12.pth 8 --eval mAP
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./tools/dist_test.sh ./configs/rotated_faster_rcnn/rotated_faster_rcnn_r50_fpn_1x_dota_le90.py ./work_dirs/rotated_faster_rcnn_r50_fpn_1x_dota_le90/epoch_12.pth 8 --format-only --eval-options submission_dir="./work_dirs/Task1_rotated_faster_rcnn_r50_fpn_1x_dota_le90_epoch_12/"
python "../DOTA_devkit/dota_evaluation_task1.py" --mergedir "./work_dirs/Task1_rotated_faster_rcnn_r50_fpn_1x_dota_le90_epoch_12/" --imagesetdir "./data/DOTA/val/" --use_07_metric True
```

## Acknowledgement

Please also support two representation learning works on which this work is based:

imTED: [paper](https://arxiv.org/abs/2205.09613) [code](https://github.com/LiewFeng/imTED)
HiViT: [paper](https://arxiv.org/abs/2205.14949) [code](https://github.com/zhangxiaosong18/hivit)

Also thanks to [Xue Yang](https://yangxue0827.github.io/) for his inspiration in the field of Oriented Object Detection.

## Citation

Not available NOW.

## License

STD is released under the [License](https://github.com/yuhongtian17/Spatial-Transform-Decoupling/blob/main/LICENSE).
