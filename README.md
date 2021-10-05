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