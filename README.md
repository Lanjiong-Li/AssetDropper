# AssetDropper

## Installation
```bash
conda create -n assetdropper python=3.10 -y
pip install -r requirements.txt
conda activate assetdropper
```

## Usage

### Prepare Input
To help you get started with your own images, you should follow this simple data structure: 
Put your own **image** (`.jpg` or `.png`) & corresponding **mask** (`.jpg` or `.png`) & **caption** in the subdirectory of data

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
    --pretrained_model_name_or_path "<huggingface_url>" \
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