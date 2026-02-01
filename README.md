# HomoDiff
Homography estimation for multimodal images with noise degradation

:star: If HomoDiff is helpful to your projects, please help star this repo. Thanks! :hugs:

## Dependencies and Installation
- Python == 3.9.20
- CUDA == 12.8
- pytorch == 2.5.0
- opencv-contrib-python == 4.10.0.84
- torchvision == 0.20.0+cu121

## Dataset
The [FLIR](https://oem.flir.com/solutions/automotive/adas-dataset-form/) dataset(RGB/Thermal)

The [HYPERSIM](https://github.com/apple/ml-hypersim) dataset(RGB/Depth)

The [RGB-NIR](https://www.epfl.ch/labs/ivrl/research/downloads/rgb-nir-scene-dataset/) Scene dataset(RGB/NIR)

## Training
### 1. Training Config
Please modify the file in [config.yml](https://github.com/Langweng/HomoDiff/blob/main/config/train_homo_flir.yaml) for different training stages
- Train from the DHE backbone
```
phase：'train'
diffusion_phase:'homo1' or 'homo2' or 'homo3
```
- Train from the AHS subnet
```
exit_train:'exit_train'
diffusion_phase:'homo1'
```

### 2. Training Command
```
CUDA_VISIBLE_DEVICES=0 python main_train.py --config /config/train_homo_flir.yaml
```

## Checkpoint
We will release our checkpoint after the paper is received.
