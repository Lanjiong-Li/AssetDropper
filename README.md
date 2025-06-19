# AssetDropper: Asset Extraction via Diffusion Models with Reward-Driven Optimization

![teaser](asset/Teaser.jpg)

### [Project Page](https://assetdropper.github.io/) | [Arxiv](https://arxiv.org/abs/2506.07738) | [Models](https://huggingface.co/LLanv/AssetDropper)

![Version](https://img.shields.io/badge/version-1.0.0-blue) &nbsp;
 <a href='https://arxiv.org/abs/2506.07738'><img src='https://img.shields.io/badge/arXiv-2506.07738-b31b1b.svg'></a> &nbsp;
 <a href='https://assetdropper.github.io/'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp;
[![HuggingFace Model](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-green)](https://huggingface.co/LLanv/AssetDropper)&nbsp;

## Installation
```bash
git clone https://github.com/Lanjiong-Li/AssetDropper.git
cd AssetDropper

conda create -n assetdropper python=3.10 -y
conda activate assetdropper

# Install torch, torchvision based on your machine configuration
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
```

## Usage

### Prepare Input
To help you get started with your own images, you should follow this simple data structure: 
Put your own **image** (`.jpg` or `.png`) & corresponding **mask** (`.jpg` or `.png`) & **caption** in the subdirectory of data.

here is an overview of data structure:

```bash
data
├── Caption/
│   └── example.txt
├── Image/
│   └── example.png 
├── Mask/
│   └── example.png 
└── example.txt (type in image names you want to process)
```

### Get Asset from Reference Image & Mask

Run the following command to get asset from the reference image:

```bash
python inference.py \
    --pretrained_model_name_or_path "LLanv/AssetDropper" \
    --data_dir "./data" \
    --output_dir "./output" \
    --txt_name "example" \
    --test_batch_size 8 \
    --guidance_scale 2.0 \
    --num_inference_steps 120 \
```
- `--pretrained_model_name_or_path`：Path to the pre-trained AssetDropper model checkpoint.  
- `--data_dir`：Path to the directory containing input images & masks.  
- `--output_dir`：Path to the output directory. 
- `--txt_name`：Name of the file that record the image name you want to process. 

or simply run:
```bash
bash inference.sh
```

## Citation
If you find this work useful for your research, please consider citing:
```
@article{li2025assetdropper,
  title={AssetDropper: Asset Extraction via Diffusion Models with Reward-Driven Optimization},
  author={Li, Lanjiong and Zhao, Guanhua and Zhu, Lingting and Cai, Zeyu and Yu, Lequan and Zhang, Jian and Wang, Zeyu},
  journal={arXiv preprint arXiv:2506.07738},
  year={2025}
}
```