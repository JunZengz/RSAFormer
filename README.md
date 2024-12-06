# RSAFormer: A method of polyp segmentation with region self-attention transformer
## Overview

## Create Environment
```
conda create -n RSAFormer python==3.8.16
conda activate RSAFormer
```

## Install Dependencies
```    
pip install -r requirements.txt
```

## Download Checkpoint 
Download pretrained checkpoints from [Google Drive](https://drive.google.com/file/d/18n1UdWEL31XN20hJDBqP0M5ccU4InnQT/view?usp=sharing) and move it to the `pretrained_pth` directory.

## Download Dataset
Download dataset from [Google Drive](https://drive.google.com/file/d/1a2jSM8cMR8huxt7VQNg1Jo4KCkZdp0TT/view?usp=sharing) and move it to the `data` directory.

## Train && Test
```
python expr.py
```

## Citation
Please cite our paper if you find the work useful:
```
@article{yin2024rsaformer,
  title={RSAFormer: A method of polyp segmentation with region self-attention transformer},
  author={Yin, Xuehui and Zeng, Jun and Hou, Tianxiao and Tang, Chao and Gan, Chenquan and Jain, Deepak Kumar and García, Salvador},
  journal={Computers in Biology and Medicine},
  volume={172},
  pages={108268},
  year={2024},
  publisher={Elsevier}
}
```

## Contact

Please contact zeng.cqupt@gamil.com for any further questions.