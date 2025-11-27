# GALIP: Generative Adversarial CLIPs for Text-to-Image Synthesis

This repository contains the code for GALIP, a Text-to-Image generation model.

## Installation

Install the required dependencies using pip:

```bash
pip install -r requirements.txt
```

## Training

The main training script is `code/src/train.py`. You can configure the training process using the YAML files located in the `code/cfg/` directory.

### How to Train

To start training, run the following command:

```bash
python code/src/train.py --cfg code/cfg/coco.yml --gpu_id 0
```

- `--cfg`: Path to the configuration file (e.g., `coco.yml`, `birds.yml`, `custom.yml`).
- `--gpu_id`: The ID of the GPU to use for training.
- Use `--multi_gpus True` for multi-GPU training.

Models and logs will be saved in the `saved_models/` and `logs/` directories, respectively.

## Evaluation with Long-CLIP

This project uses Long-CLIP for evaluating the quality of generated images.

### 1. Download Long-CLIP Weights

Before running the evaluation, you need to download the pre-trained Long-CLIP model weights.

1.  Download the weights from the [LongCLIP-B Hugging Face repository](https://huggingface.co/BeichenZhang/LongCLIP-B).
2.  Create the following directory structure inside the `Long-CLIP` folder: `Long-CLIP/checkpoints/LongCLIP-B/`.
3.  Place the downloaded model file (`longclip-B.pt`) into this directory. The final path should be `Long-CLIP/checkpoints/LongCLIP-B/longclip-B.pt`.

### 2. Run Evaluation Script

The script `code/compute_longclip_scores.py` is used to calculate the CLIP scores between generated images and their corresponding text prompts.

Before running, you may need to modify the following paths inside `code/compute_longclip_scores.py`:
- `CAPTIONS_FILE`: Path to your text file containing the captions used for generation.
- `IMAGES_ROOT`: Path to the directory containing the generated images. The images should be organized in sub-folders per caption.

Once configured, run the script:

```bash
python code/compute_longclip_scores.py
```

The script will output the average CLIP score and save a detailed `longclip_repo_scores.json` file in the root directory.
