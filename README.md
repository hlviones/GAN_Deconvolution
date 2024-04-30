# GAN_Deconvolution


Machine learning (ML) algorithms have become increasingly prevalent in image restoration use cases. An example of one of these techniques is denoising which aims to remove noise from bioimaging data, resulting in increased clarity of relevant features and structures. ML has proved to be very effective at restoring noisy images as illustrated by Noise2Void (Krull et al., 2019). Deconvolution is another essential technique, closely related to denoising. While denoising aims to remove noise from images, deconvolution is the process of attempting to remove blur caused by the imaging hardware. Similarly to denoising, deconvolution is important as it can recover details previously distorted in the input imaging data. However, there is yet to be a ML technique which manages to accomplish this with superior results and efficiency to traditional algorithm-based tools. As such, we looked to explore machine learning based alternatives to these traditional image deconvolution algorithms.

## Proposed GAN Architechture
![image](https://github.com/hlviones/GAN_Deconvolution/assets/83133751/510cd9c1-0327-4b7c-b8fd-9ee847fd8ce6) 

Our generator used the LUCYD network (https://github.com/ctom2/lucyd-deconvolution/) with some modifications such as swapping the deconvolutional layers for nearest neighbour upscaling in combinatation with a convolutional layer. 

## Dependancies
- matplotlib
- tifffile
- torch
- torchvision
- torchmetrics
- pillow
- scipy
- tqdm

## Setup

Conda enviroment can be created from the enviroment.yml file in the root directory of this project.

```
git clone https://github.com/hlviones/GAN_Deconvolution
cd GAN_Deconvolution
conda env create -f environment.yml
conda activate GAN_Deconvolution
```

## Training:

```
python GAN_Deconvolution/scripts/training.py
```


## Prediction:

```
python GAN_Deconvolution/scripts/predict.py
```
