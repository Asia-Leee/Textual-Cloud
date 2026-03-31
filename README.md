# Sight, Sound, and Sense: Unfolding Textual Clouds for Audio-Visual Generalized Zero-Shot Learning

This repository is the official implementation.
<p align="center">
  <img src="fig.jpg" alt="Framework" width="80%">
</p>

## Requirements
We recommend using **Conda** to manage your environment:
```bash
conda env create -f environment.yml
conda activate zero
```

## Step 1: Obtaining Datasets
### 1.Download GZSL benchmark datasets: VGGSound-GZSL, ActivityNet-GZSL, and UCF-GZSL
Encoded by Cip&Clap: [ Cip&Clap](https://github.com/dkurzend/ClipClap-GZSL)  

Encoded by C3D&VGGish  and SeLavi : [AVCA](https://github.com/ExplainableML/AVCA-GZSL)
### 2.Download Textual Cloud ☁️
We provide the pre-processed textual cloud data. You can choose to download it directly or generate it from scratch.

Option 1: Direct Download

You can download the pre-extracted features from [textual_cloud](https://drive.google.com/file/d/1PMt8EuDySrSMdrlDO4C8pCbPTdTa-LaF/view?usp=drive_link) and place the file in the `./data/` directory.

Option 2: Generate from Scratch

If you want to generate the textual clouds manually, please follow these steps:

1. **Obtain descriptions**: Run the following script to generate 500 semantic descriptions using Llama:
   ```bash
   python get_textual_cloud_llama.py
   ```
2. Extract Top-K clouds: Run the following script to filter and obtain the top-50 most representative textual clouds:
   ```bash
   python get_topk_textual_cloud.py
   ```
