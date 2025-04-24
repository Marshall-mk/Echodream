## Table of contents
1. [Environment setup](#environment-setup)
2. [Data preparation](#data-preparation)
3. [The models](#the-models)
4. [Training](#training)
5. [Generating EchoNet-xx](#generating-echonet-xx)
6. [Evaluation](#evaluation)
7. [Results](#results)
8. [Citation](#citation)

## Environment setup
<!-- <details open id="environment-setup">
<summary style="font-size: 1.5em; font-weight: bold;" >Environment setup<hr></summary> -->

First, we need to set up the environment. We use the following command to create a new conda environment with the required dependencies.

```bash
conda create -y -n echodream python=3.11
conda activate echodream
pip install -e .
```
*Note: the exact version of each package can be found in requirements.txt if necessary*

This repository is largely adapted from the [Echonet-synthetic repository](https://huggingface.co/HReynaud/EchoNet-Synthetic). i.e.
- The Variational Auto-Encoder (VAE) is used as is from the repo.

We also rely on external libraries to:
- evaluate the generated images and videos (StyleGAN-V)

How to install the external libraries is explained in the [External libraries](external/README.md) section.

<!-- </details> -->

## Data preparation
<!-- <details open id="data-preparation">
<summary style="font-size: 1.5em; font-weight: bold;">Data preparation<hr></summary> -->

### ➡ Original datasets
Download the EchoNet-Dynamic dataset from [here](https://echonet.github.io/dynamic/), the EchoNet-Pediatric dataset from [here](https://echonet.github.io/pediatric/), and the CardiacNet data from [here](https://www.kaggle.com/datasets/xiaoweixumedicalai/abnormcardiacechovideos). The datasets are available for free upon request. Once downloaded, extract the content of the archive in the `datasets` folder. For simplicity and consistency, we structure them like so (more info on cardiacnet below):
```
datasets
├── EchoNet-Dynamic
│   ├── Videos
│   ├── FileList.csv
│   └── VolumeTracings.csv
├── CardiacNet
│   ├── Videos
│   └── FileList.csv
└── EchoNet-Pediatric
    ├── A4C
    │   ├── Videos
    │   ├── FileList.csv
    │   └── VolumeTracings.csv
    └── PSAX
        ├── Videos
        ├── FileList.csv
        └── VolumeTracings.csv
```

To harmonize the datasets, we add some information to the `FileList.csv` files of the EchoNet-Pediatric dataset, namely FrameHeight, FrameWidth, FPS, NumberOfFrames. We also arbitrarily set the splits from the 10-fold indices to a simple TRAIN/VAL/TEST split. These updates ares applied with the following command:

```bash
python scripts/complete_pediatrics_filelist.py --dataset datasets/EchoNet-Pediatric/A4C
python scripts/complete_pediatrics_filelist.py --dataset datasets/EchoNet-Pediatric/PSAX
```

We also restructured the CardiacNet dataset to include a 'Videos' directory as well as a 'FileList.csv' file. This is done with the following commands:

```bash
python scripts/1process_cardiacnetdata.py \
	--folder_path ./CardiacNet/ \
	--output_format mp4 \
	--fps 32 \
	--resize 112,112

mkdir -p ./datasets/CardiacNet/ 
mv -r ./CardiacNet/Videos ./datasets/CardiacNet/Videos 
cp  ./CardiacNet/FileList.csv ./datasets/CardiacNet/FileList.csv

python scripts/2split_and_update_filelist.py \
	--csv_path ./datasets/CardiacNet/FileList.csv
```
This is crucial for the other scripts to work properly.

### ➡ Latent Video datasets for LVDM training

The LVDM is trained on pre-encoded latent representations of the videos. To encode the videos, we use the image VAE. You can either retrain the VAE or download it from [here](https://huggingface.co/HReynaud/EchoNet-Synthetic/tree/main/vae). Once you have the VAE, you can encode the videos with the following command:

```bash
# For the EchoNet-Dynamic dataset
python scripts/5encode_video_dataset.py \
    --model models/vae \
    --input datasets/EchoNet-Dynamic \
    --output data/latents/dynamic \
    --gray_scale
```
```bash
# For the CardiacNet dataset
python scripts/5encode_video_dataset.py \
    --model models/vae \
    --input datasets/CardiacNet \
    --output data/latents/cardiacnet \
    --gray_scale
```
```bash
# For the EchoNet-Pediatric datasets
python scripts/5encode_video_dataset.py \
    --model models/vae \
    --input datasets/EchoNet-Pediatric/A4C \
    --output data/latents/ped_a4c \
    --gray_scale

python scripts/5encode_video_dataset.py \
    --model models/vae \
    --input datasets/EchoNet-Pediatric/PSAX \
    --output data/latents/ped_psax \
    --gray_scale
```

### ➡ Validation datasets

To quantitatively evaluate the quality of the generated images and videos, we use the StyleGAN-V repo.
We cover the evaluation process in the [Evaluation](#evaluation) section.
To enable this evaluation, we need to prepare the validation datasets. We do that with the following command:

```bash
python scripts/4create_reference_dataset.py --dataset datasets/EchoNet-Dynamic --output data/reference/dynamic --frames 128
```

```bash
python scripts/4create_reference_dataset.py --dataset datasets/CardiacNet --output data/reference/cardiacnet --frames 16
```

```bash
python scripts/4create_reference_dataset.py --dataset datasets/EchoNet-Pediatric/A4C --output data/reference/ped_a4c --frames 16
```

```bash
python scripts/4create_reference_dataset.py --dataset datasets/EchoNet-Pediatric/PSAX --output data/reference/ped_psax --frames 16
```

Note that the CardiacNet and Pediatric datasets do not support 128 frames, preventing the computation of FVD_128, because there are not enough videos lasting more 4 seconds or more. We therefore only extract 16 frames per video for these datasets.

Finally, we convert the CardiacNet videos to images to enable downstream classification:

```bash
python scripts/6video_to_jpg.py \
	--video_folder ./datasets/CardiacNet/Videos \
	--output_folder ./datasets/CardiacNet/jpg
```
</details>

## The Models
<!-- <details open id="models">
<summary style="font-size: 1.5em; font-weight: bold;">The models<hr></summary> -->

![Models](ressources/models.jpg)

*Our pipeline, using our models: LVDM and VAE*


### The VAE

You can download the pretrained VAE from [here](https://huggingface.co/HReynaud/EchoNet-Synthetic/tree/main/vae)

### The LVDMS

You can download the pretrained LVDMS from [here](https://xx) or train it yourself by following the instructions in the [LVDM training](#training) section.

### Structure

The models should be structured as follows:
```
models
├── lvdm
├── lvdm_cardiacnet
└── vae
```

<!-- </details> -->
## TRAINING
<!-- <details open id="training">
<summary style="font-size: 1.5em; font-weight: bold;">Training the video models<hr></summary> -->
We trained two video models, one with text conditioning on all datasets and another with class conditioning on only the cardiacnet dataset.

```bash
# 	WITH TEXT CONDITIONING
CUDA_VISIBLE_DEVICES='0,1,2,3' accelerate launch  
	--num_processes 4  
	--multi_gpu   
	--mixed_precision fp16 
	-m  echo.lvdm.train  
	--config echo/lvdm/configs/default.yaml 
	--training_mode diffusion 
	--conditioning_type text
```
```bash
#	WITH CLASS CONDITIONING

CUDA_VISIBLE_DEVICES='0,1,2,3' accelerate launch  
	--num_processes 4  
	--multi_gpu   
	--mixed_precision fp16 
	-m  echo.lvdm.train  
	--config echo/lvdm/configs/cardiacnet.yaml 
	--training_mode diffusion 
	--conditioning_type class_id
```

<!-- </details> -->
## Generating EchoNet-xx
<!-- <details open id="echonet-xx">
<summary style="font-size: 1.5em; font-weight: bold;">Generating EchoNet-xx<hr></summary> -->

Now that we have all the necessary models, we can generate the synthetic datasets. The process is the same for all datasets and involves using the real latent images.

#### Dynamic dataset
We generate the synthetic videos with the LVDM:
```bash
CUDA_VISIBLE_DEVICES='0' python -m echo.lvdm.sample  
	--config echo/lvdm/configs/default.yaml   
	--unet ./models/lvdm/checkpoint-100000/unet_ema   
	--vae ./models/vae   
	--conditioning ./data/latents/dynamic/Latents   
	--output ./samples/lvdm_dynamic_with_text
	--num_samples 2000    
	--batch_size 16    
	--num_steps 256     
	--save_as mp4,jpg    
	--frames 192 
	--sampling_mode diffusion 
	--conditioning_type text
```

#### Cardiacnet dataset
```bash
CUDA_VISIBLE_DEVICES='0' python -m echo.lvdm.sample  
	--config echo/lvdm/configs/default.yaml   
	--unet ./models/lvdm/checkpoint-100000/unet_ema   
	--vae ./models/vae   
	--conditioning ./data/latents/cardiacnet/Latents   
	--output ./samples/lvdm_cardiac_with_text
	--num_samples 2000    
	--batch_size 16    
	--num_steps 256     
	--save_as mp4,jpg    
	--frames 192 
	--sampling_mode diffusion 
	--conditioning_type text
```

#### pediatric datasets
```bash
CUDA_VISIBLE_DEVICES='0' python -m echo.lvdm.sample  
	--config echo/lvdm/configs/default.yaml   
	--unet ./models/lvdm/checkpoint-100000/unet_ema   
	--vae ./models/vae   
	--conditioning ./data/latents/ped_a4c/Latents   
	--output ./samples/lvdm_ped_a4c_with_text
	--num_samples 2000    
	--batch_size 16    
	--num_steps 256     
	--save_as mp4,jpg    
	--frames 192 
	--sampling_mode diffusion 
	--conditioning_type text

CUDA_VISIBLE_DEVICES='0' python -m echo.lvdm.sample  
	--config echo/lvdm/configs/default.yaml   
	--unet ./models/lvdm/checkpoint-100000/unet_ema   
	--vae ./models/vae   
	--conditioning ./data/latents/ped_psax/Latents   
	--output ./samples/lvdm_ped_psax_with_text
	--num_samples 2000    
	--batch_size 16    
	--num_steps 256     
	--save_as mp4,jpg    
	--frames 192 
	--sampling_mode diffusion 
	--conditioning_type text
```
We can use the class conditioned lvdm model to generate the Cardiac-Synthetic dataset using:

```bash
CUDA_VISIBLE_DEVICES='0' python -m echo.lvdm.sample  
	--config echo/lvdm/configs/cardiacnet.yaml   
	--unet ./models/lvdm_cardiacnet/checkpoint-100000/unet_ema   
	--vae ./models/vae   
	--conditioning ./data/latents/cardiacnet/Latents   
	--output ./samples/lvdm_cardiacnet  
	--num_samples 655    
	--batch_size 16    
	--num_steps 256     
	--save_as mp4,jpg    
	--frames 192 
	--sampling_mode diffusion 
	--conditioning_type class_id 
	--class_ids 4
```

Finally, we prepare the synthetic data for downstream evaluation
```bash
mkdir -p ./datasets/Cardiac-synthetic/
cp -r ./samples/lvdm_cardiacnet/jpg  ./datasets/Cardiac-synthetic/jpg
cp ./samples/lvdm_cardiacnet/FileList.csv ./datasets/Cardiac-synthetic/FileList.csv

python scripts/update_split_filelist.py --csv ./datasets/Cardiac-synthetic/FileList.csv
 	--train 457 
	--val 97 
	--test 101

python scripts/update_synt_file.py 
	--input ./datasets/Cardiac-synthetic/FileList.csv 
	--output ./datasets/Cardiac-synthetic/FileList.csv
```

<!-- </details> -->

## Evaluation
<!-- <details open id="evaluation">
<summary style="font-size: 1.5em; font-weight: bold;">Evaluation<hr></summary> -->
We evaluate the generative performance of both the class-conditioned and text-conditioned models using the StyleGAN-V library

```bash	
cd external/stylegan-v
```

```bash	
# WITH TEXT CONDITIONING
CUDA_VISIBLE_DEVICES='0' python src/scripts/calc_metrics_for_dataset.py
	--real_data_path ./data/reference/cardiacnet
	--fake_data_path ./samples/lvdm_cardiac_with_text/jpg 
	--mirror 0 --gpus 1 --resolution 112  
	--metrics fvd2048_16f,fid50k_full,is50k >> "./samples/lvdm_cardiac_with_text/metrics.txt"

CUDA_VISIBLE_DEVICES='0' python src/scripts/calc_metrics_for_dataset.py
	--real_data_path ./data/reference/ped_psax
	--fake_data_path ./samples/lvdm_ped_psax_with_text/jpg 
	--mirror 0 --gpus 1 --resolution 112  
	--metrics fvd2048_16f,fid50k_full,is50k >> "./samples/lvdm_ped_psax_with_text/metrics.txt"

CUDA_VISIBLE_DEVICES='0' python src/scripts/calc_metrics_for_dataset.py
	--real_data_path ./data/reference/ped_a4c
	--fake_data_path ./samples/lvdm_ped_a4c_with_text/jpg 
	--mirror 0 --gpus 1 --resolution 112  
	--metrics fvd2048_16f,fid50k_full,is50k >> "./samples/lvdm_ped_a4c_with_text/metrics.txt"

CUDA_VISIBLE_DEVICES='0' python src/scripts/calc_metrics_for_dataset.py
	--real_data_path ./data/reference/dynamic
	--fake_data_path ./samples/lvdm_dynamic_with_text/jpg 
	--mirror 0 --gpus 1 --resolution 112  
	--metrics fvd2048_16f,fvd2048_128f,fid50k_full,is50k >> "./samples/lvdm_dynamic_with_text/metrics.txt"
```		

```bash		
# WITH CLASS CONDITIONING
CUDA_VISIBLE_DEVICES='0' python src/scripts/calc_metrics_for_dataset.py
	--real_data_path ./data/reference/cardiacnet
	--fake_data_path ./samples/lvdm_cardiacnet/jpg 
	--mirror 0 --gpus 1 --resolution 112  
	--metrics fvd2048_16f,fid50k_full,is50k >> "./samples/lvdm_cardiacnet/metrics.txt"
```

We evaluate the class-conditioned model on a classification task. We do these following the steps below:
