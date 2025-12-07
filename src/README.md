<a name="readme-top"></a>

<div align="center">
  <h1>📦 Source Code</h1>
  <p>
    Core implementation of the Tiny Diffusion model and StepDrop sampling components
  </p>
  <p>
    <a href="#quick-start"><strong>Quick Start »</strong></a>
    ·
    <a href="#training">Training</a>
    ·
    <a href="#sampling">Sampling</a>
    ·
    <a href="#custom-datasets">Custom Datasets</a>
  </p>
</div>

---

<details>
  <summary>📑 Table of Contents</summary>
  <ol>
    <li><a href="#about">About</a></li>
    <li><a href="#file-structure">File Structure</a></li>
    <li><a href="#quick-start">Quick Start</a></li>
    <li><a href="#training">Training</a></li>
    <li><a href="#sampling">Sampling</a></li>
    <li><a href="#custom-datasets">Custom Datasets</a></li>
    <li><a href="#configuration">Configuration</a></li>
    <li><a href="#cli-reference">CLI Reference</a></li>
    <li><a href="#examples">Examples</a></li>
  </ol>
</details>

---

## About

This directory contains the modular, CLI-friendly implementation of the **Tiny Diffusion** model with support for:

- 🎯 **Multiple datasets**: MNIST, CIFAR-10, and custom image folders
- ⚡ **Fast sampling**: DDPM and DDIM samplers
- 🔧 **Flexible configuration**: CLI arguments or JSON config files
- 📊 **Evaluation metrics**: FID, Inception Score, and FLOPs

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## File Structure

### Core Modules

| File | Description |
|:-----|:------------|
| `config.py` | 🔧 Configuration management with CLI argument parsing and JSON support |
| `dataset.py` | 📂 Data loading for MNIST, CIFAR-10, and custom image datasets |
| `modules.py` | 🧠 U-Net architecture with ResNet blocks, attention, and time embeddings |
| `scheduler.py` | 📈 Noise schedules (linear, cosine) and diffusion process variables |
| `train.py` | 🏋️ Training script with checkpointing, logging, and LR scheduling |
| `sample.py` | 🎨 DDPM and DDIM sampling with grid/individual image output |

### Evaluation

| File | Description |
|:-----|:------------|
| `eval/metrics_utils.py` | 📊 FID, Inception Score, and FLOPs computation utilities |

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Quick Start

### Training

```bash
# Train on MNIST (default)
python train.py

# Train on CIFAR-10
python train.py --dataset cifar10 --epochs 50

# Train on custom dataset
python train.py \
    --dataset custom \
    --custom_data_dir /path/to/your/images \
    --img_size 64 \
    --channels 3 \
    --epochs 100
```

### Sampling

```bash
# DDPM sampling (high quality)
python sample.py --checkpoint checkpoints/model.pt

# DDIM sampling (fast)
python sample.py \
    --checkpoint checkpoints/model.pt \
    --method ddim \
    --ddim_steps 50
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Training

### Basic Usage

```bash
python train.py [OPTIONS]
```

### Training Options

| Argument | Default | Description |
|:---------|:--------|:------------|
| `--dataset` | `mnist` | Dataset: `mnist`, `cifar10`, `custom` |
| `--custom_data_dir` | `None` | Path to custom images folder |
| `--img_size` | `28` | Image size (height = width) |
| `--channels` | `1` | Number of image channels |
| `--batch_size` | `128` | Training batch size |
| `--epochs` | `20` | Number of training epochs |
| `--lr` | `2e-4` | Learning rate |
| `--n_timesteps` | `1000` | Number of diffusion timesteps |
| `--schedule_type` | `cosine` | Noise schedule: `linear`, `cosine` |
| `--base_channels` | `64` | U-Net base channel count |
| `--save_path` | `checkpoints/model.pt` | Model save path |
| `--log_dir` | `logs` | Directory for training logs |
| `--seed` | `None` | Random seed for reproducibility |
| `--resume` | `None` | Path to checkpoint to resume from |
| `--config` | `None` | Path to JSON config file |

### Resume Training

```bash
python train.py --resume checkpoints/model.pt
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Sampling

### Basic Usage

```bash
python sample.py --checkpoint <path_to_model> [OPTIONS]
```

### Sampling Options

| Argument | Default | Description |
|:---------|:--------|:------------|
| `--checkpoint` | **required** | Path to trained model |
| `--method` | `ddpm` | Sampling method: `ddpm`, `ddim` |
| `--n_samples` | `16` | Number of samples to generate |
| `--ddim_steps` | `50` | DDIM sampling steps (faster = fewer) |
| `--ddim_eta` | `0.0` | DDIM stochasticity (0 = deterministic) |
| `--output_dir` | `samples` | Output directory for images |
| `--save_grid` | `True` | Save samples as image grid |
| `--save_individual` | `False` | Save each sample individually |
| `--show` | `False` | Display samples (requires display) |

### Sampling Methods Comparison

| Method | Steps | Speed | Quality |
|:-------|:------|:------|:--------|
| **DDPM** | 1000 | 🐢 Slow | ⭐⭐⭐ Best |
| **DDIM** (50 steps) | 50 | 🚀 Fast | ⭐⭐ Great |
| **DDIM** (25 steps) | 25 | ⚡ Very Fast | ⭐ Good |

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Custom Datasets

### Folder Structure

Organize your images in a folder (subfolders are supported):

```
my_dataset/
├── image1.png
├── image2.jpg
├── subfolder/
│   ├── image3.png
│   └── image4.jpg
└── another_folder/
    └── image5.webp
```

### Supported Formats

| Format | Extensions |
|:-------|:-----------|
| PNG | `.png` |
| JPEG | `.jpg`, `.jpeg` |
| BMP | `.bmp` |
| WebP | `.webp` |
| TIFF | `.tiff` |

### Training with Custom Data

```bash
python train.py \
    --dataset custom \
    --custom_data_dir ./my_dataset \
    --img_size 64 \
    --channels 3 \
    --epochs 100
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Configuration

### Via Command Line

```bash
python train.py --lr 1e-4 --batch_size 64 --epochs 100
```

### Via JSON Config File

Create a JSON configuration file:

```json
{
    "dataset": "cifar10",
    "img_size": 32,
    "channels": 3,
    "batch_size": 128,
    "epochs": 100,
    "lr": 2e-4,
    "base_channels": 128,
    "schedule_type": "cosine"
}
```

Use it with:

```bash
python train.py --config my_config.json
```

> 💡 **Tip**: CLI arguments override config file values, so you can use a base config and customize specific parameters.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## CLI Reference

### `train.py`

```bash
python train.py --help
```

```
Usage: train.py [OPTIONS]

Train a Tiny Diffusion Model

Options:
  --dataset          Dataset to use (mnist, cifar10, custom)
  --batch_size       Training batch size
  --epochs           Number of training epochs
  --lr               Learning rate
  --save_path        Path to save model checkpoint
  --resume           Resume from checkpoint
  --config           Load configuration from JSON file
  ...
```

### `sample.py`

```bash
python sample.py --help
```

```
Usage: sample.py --checkpoint <path> [OPTIONS]

Generate samples from a trained diffusion model

Options:
  --checkpoint       Path to trained model (required)
  --method           Sampling method (ddpm, ddim)
  --n_samples        Number of samples to generate
  --ddim_steps       DDIM sampling steps
  --output_dir       Output directory for images
  ...
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Examples

### 1️⃣ Quick Test on MNIST

```bash
python train.py --epochs 5
python sample.py --checkpoint checkpoints/model.pt
```

### 2️⃣ Train on CIFAR-10 with Larger Model

```bash
python train.py \
    --dataset cifar10 \
    --epochs 100 \
    --base_channels 128 \
    --batch_size 64
```

### 3️⃣ Fast Sampling with DDIM

```bash
python sample.py \
    --checkpoint checkpoints/model.pt \
    --method ddim \
    --ddim_steps 25 \
    --n_samples 64
```

### 4️⃣ Train on Your Own Photos

```bash
python train.py \
    --dataset custom \
    --custom_data_dir ~/my_photos \
    --img_size 128 \
    --channels 3 \
    --batch_size 32 \
    --epochs 200
```

### 5️⃣ Resume Interrupted Training

```bash
python train.py --resume checkpoints/checkpoint_epoch_50.pt
```

### 6️⃣ Generate Many Samples

```bash
python sample.py \
    --checkpoint checkpoints/model.pt \
    --n_samples 100 \
    --method ddim \
    --ddim_steps 50 \
    --output_dir my_samples/ \
    --save_individual
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

<div align="center">
  <p>
    <a href="../README.md">← Back to Main README</a>
  </p>
</div>