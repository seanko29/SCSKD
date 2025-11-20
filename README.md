# SCSKD - CVPR2026 Submission 



### Student-Conditioned Spectral Knowledge Distillation for Image Super-Resolution




## Install
Create a conda enviroment:
````
ENV_NAME="scskd"
conda create -n $ENV_NAME python=3.10
conda activate $ENV_NAME
````
Run following script to install the dependencies:
````
bash install.sh
````


## Usage
Pre-trained checkpoints and visual results can be downloaded [here](https://drive.google.com/drive/folders/1NdJ1D0nNxwJWxgEKMwlIuzgGVYhLrKW-?usp=sharing). Place the checkpoints in `checkpoints/` or `pretrained/`.

In `options` you can find the corresponding config files for reproducing our experiments.

##### **Testing**
For testing the pre-trained checkpoints please use following commands. Replace `[TEST OPT YML]` with the path to the corresponding option file.
`````
CUDA_VISIBLE_DEVICES=0 python basicsr/test.py -opt [TEST OPT YML]
`````

##### **Training**
For single-GPU training use the following commands. Replace `[TRAIN OPT YML]` with the path to the corresponding option file.
`````
CUDA_VISBILE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=4321 basicsr/train.py -opt [TRAIN OPT YML] --launcher pytorch

Example.
CUDA_VISBILE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=4321 basicsr/train.py -opt train_scskd_edsr_x4.yml --launcher pytorch

`````
For multiple-GPU training (ex. 2 A100s), use the following commands. Replace `[TRAIN OPT YML]` with the path to the corresponding option file.
`````
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=4321 basicsr/train.py -opt [TRAIN OPT YML] --launcher pytorch


Example.
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=4321 basicsr/train.py -opt train_scskd_edsr_x4.yml --launcher pytorch
`````
## Acknowledgements

This code is built on [BasicSR](https://github.com/XPixelGroup/BasicSR).
