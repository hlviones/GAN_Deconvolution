<p >
    <img src="https://zenodo.org/badge/DOI/10.5281/zenodo.11073626.svg"
        height="20">
</p>

# GAN_Deconvolution

## Example results

![image](https://github.com/hlviones/GAN_Deconvolution/assets/83133751/7113681f-1b1a-4e11-ae45-520354ff7c2b)


Datasets used can be found at https://zenodo.org/records/11073626.

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
