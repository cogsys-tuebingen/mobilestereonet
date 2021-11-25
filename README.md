# MobileStereoNet
![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)    

This repository contains the code for "MobileStereoNet: Towards Lightweight Deep Networks for Stereo Matching" [[arXiv](https://arxiv.org/pdf/2108.09770.pdf)]
[[project](https://uni-tuebingen.de/fakultaeten/mathematisch-naturwissenschaftliche-fakultaet/fachbereiche/informatik/lehrstuehle/kognitive-systeme/projects/deepstereovision/)].

<div align="center">
    <img align="center" src="images/000005_10.png" alt="drawing" width="621"/>
    <p> <b>Input image</b> </p>
    <img src="images/000005_10_pred_2D.png" alt="drawing" width="621"/>
    <img src="images/000005_10_error_2D.png" alt="drawing" width="621"/>
    <p> <b>2D-MobileStereoNet prediction</b> </p>
    <img src="images/000005_10_pred_3D.png" alt="drawing" width="621"/>
    <img src="images/000005_10_error_3D.png" alt="drawing" width="621"/>
    <p> <b>3D-MobileStereoNet prediction</b> </p>
</div>

## Evaluation Results 
MobileStereoNets are trained and tested using [SceneFlow](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html) (SF), [KITTI](http://www.cvlibs.net/datasets/kitti/eval_stereo.php) and [DrivingStereo](https://drivingstereo-dataset.github.io/) (DS) datasets.
In the following tables, the first columns show the training sets. For instance, in the case of "SF + KITTI2015", the model
is firstly pretrained on the SceneFlow dataset, and then fine-tuned on KITTI images.
The results are reported in End-point Error (EPE); the lower, the better.   
Note that some experiments evaluate the zero-shot cross-dataset generalizability, e.g., when the model is trained on "SF + DS" and evaluated on "KITTI2015 val" or "KITTI2012 train".   
The related trained models are provided in the tables as hyperlinks.  
...

- 2D-MobileStereoNet  

|                     | SF test | DS test       | KITTI2015 val | KITTI2012 train |
|:-------------------:|:-------:|:-------------:|:-------:|:---------------:|
| SF                  | **1.14**|    6.59       |    2.42 |       2.45      |
| DS                  |    -    |    **0.67**   |    1.02 |       0.96      |
| SF + DS             |    -    |    0.73       |    1.04 |       1.04      |
| SF + KITTI2015      |    -    |    1.41       |    0.79 |       1.18      |
| DS + KITTI2015      |    -    |    0.79       |    **0.65** |       0.91      |
| SF + DS + KITTI2015 |    -    |    0.83       |    0.68  |       **0.90**  |

- 3D-MobileStereoNet  

|                     | SF test | DS test |  KITTI2015 val| KITTI2012 train |
|:-------------------:|:-------:|:-------------:|:-------:|:---------------:|
| SF                  | **0.80**|    4.50    |      10.30 |       9.38      |
| DS                  |    -    |     0.60      |     1.16|       1.14      |
| SF + DS             |    -    |   **0.57**       | 1.12 |       1.10      |
| SF + KITTI2015      |    -    |  1.53        |    0.65  |       0.90      |
| DS + KITTI2015      |    -    |   0.65      |     0.60  |       0.85      |
| SF + DS + KITTI2015 |    -    |   0.62  |     **0.59**  |     **0.83**    |

## Results on KITTI 2015 Leaderboard
[Leaderboard](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo)  
[2D-MobileStereoNet on the leaderboard](http://www.cvlibs.net/datasets/kitti/eval_scene_flow_detail.php?benchmark=stereo&result=53b8257acf35d19410db728c95e5e666890c5e27)  
[3D-MobileStereoNet on the leaderboard](http://www.cvlibs.net/datasets/kitti/eval_scene_flow_detail.php?benchmark=stereo&result=a1fd814aa8b2353df689233fb00bdaa227f380a8)

## Computational Complexity

...

## Installation

### Requirements
The code is tested on:
- Ubuntu 18.04
- Python 3.6 
- PyTorch 1.4.0 
- Torchvision 0.5.0
- CUDA 10.0

### Setting up the environment

```shell
conda env create --file mobilestereonet.yml
conda activate mobilestereonet
```

### Training 

Set a variable (e.g. ```DATAPATH```) for the dataset directory
```DATAPATH="/Datasets/SceneFlow/"```
or
```DATAPATH="/Datasets/KITTI2015/"```. Then, you can run the ```train.py``` file as below:

#### Pretraining on SceneFlow

```shell
python train.py --dataset sceneflow --datapath $DATAPATH --trainlist ./filenames/sceneflow_train.txt --testlist ./filenames/sceneflow_test.txt --epochs 20 --lrepochs "10,12,14,16:2" --batch_size 8 --test_batch_size 8 --model MSNet2D
```

#### Finetuning on KITTI

```shell
python train.py --dataset kitti --datapath $DATAPATH --trainlist ./filenames/kitti15_train.txt --testlist ./filenames/kitti15_val.txt --epochs 400 --lrepochs "200:10" --batch_size 8 --test_batch_size 8 --loadckpt ./checkpoints/pretrained.ckpt --model MSNet2D
```

The arguments in both cases can be set differently depending on the model and the system.

### Prediction

The following script creates disparity maps for a specified model:

```shell
python prediction.py --datapath $DATAPATH --testlist ./filenames/kitti15_test.txt --loadckpt ./checkpoints/finetuned.ckpt --dataset kitti --colored True --model MSNet2D
```

## Credits

The implementation of this code is based on [PSMNet](https://github.com/JiaRenChang/PSMNet) and [GwcNet](https://github.com/xy-guo/GwcNet). Also, thanks to Matteo Poggi for the [KITTI python utils](https://github.com/mattpoggi/kitti-utilities-python).

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Citation

If you use this code, please cite this project.

```
@article{shamsafar2021mobilestereonet,
  title={MobileStereoNet: Towards Lightweight Deep Networks for Stereo Matching},
  author={Shamsafar, Faranak and Woerz, Samuel and Rahim, Rafia and Zell, Andreas},
  journal={arXiv preprint arXiv:2108.09770},
  year={2021}
}
```

## Contact
The repository is maintained by [Faranak Shamsafar](https://www.linkedin.com/in/faranak-shamsafar/).  
[f.shmsfr@gmail.com](f.shmsfr@gmail.com)  