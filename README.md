

<p align="center">
  <img height="150" src="./miscellaneous/active-3d-logo.png" />
</p>

-----------------------
![visitor badge](https://visitor-badge.laobi.icu/badge?page_id=Luoyadan.KECOR-active-3Ddet&left_color=red&right_color=green&left_text=Hello%20Visitors)





This work is the official Pytorch implementation of our ICCV publication: **KECOR: Kernel Coding Rate Maximization for Active 3D Object Detection**.


[[arXiv]](https://arxiv.org/abs/2301.09249) 

## News
:fire: 08/13 updates: under development. Checkpoints will be uploaded soon. 

:fire: 03/18 updates: The checkpoints can be found via [https://drive.google.com/drive/folders/1xbEI3tSfTCHIt3m8tk4hAy4qBJ55NuqL?usp=sharing](https://drive.google.com/drive/folders/1xbEI3tSfTCHIt3m8tk4hAy4qBJ55NuqL?usp=sharing)


## Citation
```
@inproceedings{DBLP:conf/iccv/LuoCF0BH23,
  author       = {Yadan Luo and
                  Zhuoxiao Chen and
                  Zhen Fang and
                  Zheng Zhang and
                  Mahsa Baktashmotlagh and
                  Zi Huang},
  title        = {Kecor: Kernel Coding Rate Maximization for Active 3D Object Detection},
  booktitle    = {{IEEE/CVF} International Conference on Computer Vision, {ICCV} 2023, Paris, France, October 1-6, 2023},
  pages        = {18233--18244},
  publisher    = {{IEEE}},
  year         = {2023}
}
```

## Framework
Achieving a reliable LiDAR-based object detector in autonomous driving is paramount, but its success hinges on obtaining large amounts of precise 3D annotations. Active learning (AL) seeks to mitigate the annotation burden through algorithms that use fewer labels and can attain performance comparable to fully supervised learning. Although AL has shown promise, current approaches prioritize the selection of unlabeled point clouds with high aleatoric and/or epistemic uncertainty, leading to the selection of more instances for labeling and reduced computational efficiency. In this paper, we resort to a novel kernel coding rate maximization (KECOR) strategy which aims to identify the most informative point clouds to acquire labels through the lens of information theory. Greedy search is applied to seek desired point clouds that can maximize the minimal number of bits required to encode the latent features. To determine the uniqueness and informativeness of the selected samples from the model perspective, we construct a proxy network of the 3D detector head and compute the outer product of Jacobians from all proxy layers to form the empirical neural tangent kernel (NTK) matrix. To accommodate both one-stage (i.e., SECOND) and two-stage detectors (i.e., PV-RCNN), we further incorporate the classification entropy maximization and well trade-off between detection performance and the total number of bounding boxes selected for annotation. Extensive experiments conducted on two 3D benchmarks and a 2D detection dataset evidence the superiority and versatility of the proposed approach. Our results show that approximately 44\% box-level annotation costs and 26\% computational time are reduced compared to the state-of-the-art AL method, without compromising detection performance.





----
## Contents
* [Installation](#Installation)
  * [Requirements](#Requirements)
  * [pcdet v0.5](#install-pcdet-v05)
* [Getting Started](#getting-started)
  * [Requirements](#Requirements)
  * [KITTI Dataset](#KITTI-Dataset)
  * [Waymo Open Dataset](#Waymo-Open-Dataset)
  <!-- * [Lyft Dataset](#Lyft-Dataset) -->
  * [Training & Testing](#training--testing)


# Installation

### Requirements
All the codes are tested in the following environment:
* Python 3.6+
* PyTorch 1.10.1
* CUDA 11.3 
* wandb 0.12.11
* [`spconv-cu113 v2.1.21`](https://github.com/traveller59/spconv)


### Install `pcdet v0.5`
Our implementations of 3D detectors are based on the lastest [`OpenPCDet`](https://github.com/open-mmlab/OpenPCDet). To install this `pcdet` library and its dependent libraries, please run the following command:

```shell
python setup.py develop
```
> NOTE: Please re-install even if you have already installed pcdet previoursly.


# Getting Started
The **active learning configs** are located at [tools/cfgs/active-kitti_models](./tools/cfgs/active-kitti_models) and [/tools/cfgs/active-waymo_models](./tools/cfgs/active-waymo_models) for different AL methods. The dataset configs are located within [tools/cfgs/dataset_configs](./tools/cfgs/dataset_configs), 
and the model configs are located within [tools/cfgs](./tools/cfgs) for different datasets. 


## Dataset Preparation
Currently we provide the dataloader of KITTI dataset and Waymo dataset, and the supporting of more datasets are on the way.  

### KITTI Dataset
* Please download the official [KITTI 3D object detection](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) dataset and organize the downloaded files as follows (the road planes could be downloaded from [[road plane]](https://drive.google.com/file/d/1d5mq0RXRnvHPVeKx6Q612z0YRO1t2wAp/view?usp=sharing), which are optional for data augmentation in the training):
* If you would like to train [CaDDN](../tools/cfgs/kitti_models/CaDDN.yaml), download the precomputed [depth maps](https://drive.google.com/file/d/1qFZux7KC_gJ0UHEg-qGJKqteE9Ivojin/view?usp=sharing) for the KITTI training set
* NOTE: if you already have the data infos from `pcdet v0.1`, you can choose to use the old infos and set the DATABASE_WITH_FAKELIDAR option in tools/cfgs/dataset_configs/kitti_dataset.yaml as True. The second choice is that you can create the infos and gt database again and leave the config unchanged.

```
KECOR-active-3Ddet
├── data
│   ├── kitti
│   │   │── ImageSets
│   │   │── training
│   │   │   ├──calib & velodyne & label_2 & image_2 & (optional: planes) & (optional: depth_2)
│   │   │── testing
│   │   │   ├──calib & velodyne & image_2
├── pcdet
├── tools
```

* Generate the data infos by running the following command: 
```python 
python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml
```

<!-- ### NuScenes Dataset
* Please download the official [NuScenes 3D object detection dataset](https://www.nuscenes.org/download) and 
organize the downloaded files as follows: 
```
KECOR-active-3Ddet
├── data
│   ├── nuscenes
│   │   │── v1.0-trainval (or v1.0-mini if you use mini)
│   │   │   │── samples
│   │   │   │── sweeps
│   │   │   │── maps
│   │   │   │── v1.0-trainval  
├── pcdet
├── tools
```

* Install the `nuscenes-devkit` with version `1.0.5` by running the following command: 
```shell script
pip install nuscenes-devkit==1.0.5
```

* Generate the data infos by running the following command (it may take several hours): 
```python 
python -m pcdet.datasets.nuscenes.nuscenes_dataset --func create_nuscenes_infos \
    --cfg_file tools/cfgs/dataset_configs/nuscenes_dataset.yaml \
    --version v1.0-trainval
``` -->

### Waymo Open Dataset
* Please download the official [Waymo Open Dataset](https://waymo.com/open/download/), 
including the training data `training_0000.tar~training_0031.tar` and the validation 
data `validation_0000.tar~validation_0007.tar`.
* Unzip all the above `xxxx.tar` files to the directory of `data/waymo/raw_data` as follows (You could get 798 *train* tfrecord and 202 *val* tfrecord ):  
```
KECOR-active-3Ddet
├── data
│   ├── waymo
│   │   │── ImageSets
│   │   │── raw_data
│   │   │   │── segment-xxxxxxxx.tfrecord
|   |   |   |── ...
|   |   |── waymo_processed_data_v0_5_0
│   │   │   │── segment-xxxxxxxx/
|   |   |   |── ...
│   │   │── waymo_processed_data_v0_5_0_gt_database_train_sampled_1/
│   │   │── waymo_processed_data_v0_5_0_waymo_dbinfos_train_sampled_1.pkl
│   │   │── waymo_processed_data_v0_5_0_gt_database_train_sampled_1_global.npy (optional)
│   │   │── waymo_processed_data_v0_5_0_infos_train.pkl (optional)
│   │   │── waymo_processed_data_v0_5_0_infos_val.pkl (optional)
├── pcdet
├── tools
```
* Install the official `waymo-open-dataset` by running the following command: 
```shell script
pip3 install --upgrade pip
pip3 install waymo-open-dataset-tf-2-0-0==1.2.0 --user
```
> Waymo version in our project is 1.2.0

* Extract point cloud data from tfrecord and generate data infos by running the following command (it takes several hours, 
and you could refer to `data/waymo/waymo_processed_data_v0_5_0` to see how many records that have been processed): 
```python 
python -m pcdet.datasets.waymo.waymo_dataset --func create_waymo_infos \
    --cfg_file tools/cfgs/dataset_configs/waymo_dataset.yaml
```

Note that you do not need to install `waymo-open-dataset` if you have already processed the data before and do not need to evaluate with official Waymo Metrics. 



## Training & Testing


### Test and evaluate the pretrained models
The weights of our pre-trained model will be released upon acceptance.

* Test with a pretrained model: 
```shell script
python test.py --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE} --ckpt ${CKPT}
```

* To test all the saved checkpoints of a specific training setting and draw the performance curve on the Tensorboard, add the `--eval_all` argument: 
```shell script
python test.py --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE} --eval_all
```

* To test with multiple GPUs:
```shell script
sh scripts/dist_test.sh ${NUM_GPUS} \
    --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE}

# or

sh scripts/slurm_test_mgpu.sh ${PARTITION} ${NUM_GPUS} \
    --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE}
```

### Train a backbone
In our active learning setting, the 3D detector will be pre-trained with a small labeled set $\mathcal{D}_L$ which is randomly sampled from the trainig set. To train such a backbone, please run

```shell script
sh scripts/${DATASET}/train_${DATASET}_backbone.sh
```


### Train with different active learning strategies
We provide several options for active learning algorithms, including
- random selection [`random`]
- confidence sample [`confidence`]
- entropy sampling [`entropy`]
- MC-Reg sampling [`montecarlo`]
- greedy coreset [`coreset`]
- learning loss [`llal`]
- BADGE sampling [`badge`]
- CRB sampling [`crb`]
- Kecor sampling [`kecor`]


You could optionally add extra command line parameters `--batch_size ${BATCH_SIZE}` and `--epochs ${EPOCHS}` to specify your preferred parameters. 

* Train:
```shell script
python train.py --cfg_file ${CONFIG_FILE}
```
### Acknowledgement

Part of code for NTK implementation is from [https://github.com/dholzmueller/bmdal_reg](https://github.com/dholzmueller/bmdal_reg)
