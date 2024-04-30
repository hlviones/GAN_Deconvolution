<p >
    <img src="https://zenodo.org/badge/DOI/10.5281/zenodo.11073626.svg"
        height="20">
</p>

# GAN_Deconvolution


Machine learning (ML) algorithms have become increasingly prevalent in image restoration use cases. An example of one of these techniques is denoising which aims to remove noise from bioimaging data, resulting in increased clarity of relevant features and structures. ML has proved to be very effective at restoring noisy images as illustrated by Noise2Void (Krull et al., 2019). Deconvolution is another essential technique, closely related to denoising. While denoising aims to remove noise from images, deconvolution is the process of attempting to remove blur caused by the imaging hardware. Similarly to denoising, deconvolution is important as it can recover details previously distorted in the input imaging data. However, there is yet to be a ML technique which manages to accomplish this with superior results and efficiency to traditional algorithm-based tools. As such, we looked to explore machine learning based alternatives to these traditional image deconvolution algorithms.

## Example results

![image](https://github.com/hlviones/GAN_Deconvolution/assets/83133751/7113681f-1b1a-4e11-ae45-520354ff7c2b)

### Qualitative comparison between ML tool deconvolution of U2OS cells

Input data is image of U2OS cells acquired by widefield microscopy from the Li et al dataset (A) (Li et al., 2022). LUCYD deconvolution recovers several features and structures both intra and extracellularly (B). While some very slight patch edge artifacts are visible after LUCYD deconvolution, the general clarity of the cells and their surrounding environment was greatly improved. (B) When an individual cell was closely examined the intra and extra cellular structures appear much clearer post-deconvolution. (C) Following GAN deconvolution, the clarity of the intracellular and extracellular structures was further improved. There were also no visible patch edge artifacts distorting the final output image

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
