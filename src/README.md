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
    <a href="#stepdrop-samplers">StepDrop Samplers</a>
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
    <li><a href="#stepdrop-samplers">StepDrop Samplers</a></li>
    <li><a href="#custom-datasets">Custom Datasets</a></li>
    <li><a href="#configuration">Configuration</a></li>
    <li><a href="#cli-reference">CLI Reference</a></li>
    <li><a href="#examples">Examples</a></li>
  </ol>
</details>

---

## About

This directory contains the modular, CLI-friendly implementation of the **Tiny Diffusion** model with support for:

- 🎯 Multiple datasets: MNIST, CIFAR-10, and custom image folders
- ⚡ Fast sampling: DDPM, DDIM, and StepDrop samplers
- 🎲 StepDrop strategies: Linear, Quadratic, Cosine², Adaptive, and more
- 🔧 Flexible configuration: CLI arguments or JSON config files
- 📊 Evaluation metrics: FID, Inception Score, LPIPS, and more

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## File Structure

### Core Modules

| File | Description |
|:-----|:------------|
| `src/config.py` | Configuration management with CLI argument parsing and JSON support |
| `src/dataset.py` | Data loading for MNIST, CIFAR-10, and custom image datasets |
| `src/modules.py` | U-Net architecture with ResNet blocks, attention, and time embeddings |
| `src/scheduler.py` | Noise schedules (linear, cosine) and diffusion process variables |
| `src/train.py` | Training script with checkpointing, logging, and LR scheduling |
| `src/sample.py` | DDPM, DDIM, and StepDrop sampling with grid/individual image output |

### Sampler Package (`src/sampler/`)

| File | Description |
|:-----|:------------|
| `__init__.py` | Package exports for easy importing |
| `DDPM.py` | Standard DDPM sampler (1000 steps) |
| `DDIM.py` | Accelerated DDIM sampler (configurable steps) |
| `stepdrop.py` | StepDrop and AdaptiveStepDrop samplers |

### Evaluation (`src/eval/`)

| File | Description |
|:-----|:------------|
| `metrics_utils.py` | FID, KID, IS, LPIPS, Precision/Recall, and more |

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
# DDPM sampling (high quality, slow)
python sample.py --checkpoint checkpoints/model.pt

# DDIM sampling (fast)
python sample.py \
  --checkpoint checkpoints/model.pt \
  --method ddim \
  --ddim_steps 50

# StepDrop sampling (adaptive speed/quality)
python sample.py \
  --checkpoint checkpoints/model.pt \
  --method stepdrop \
  --skip_prob 0.3 \
  --skip_strategy linear
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
| `--checkpoint` | required | Path to trained model |
| `--method` | `ddpm` | Sampling method: `ddpm`, `ddim`, `stepdrop`, `adaptive_stepdrop` |
| `--n_samples` | `16` | Number of samples to generate |
| `--ddim_steps` | `50` | DDIM sampling steps |
| `--ddim_eta` | `0.0` | DDIM stochasticity (0 = deterministic) |
| `--skip_prob` | `0.3` | StepDrop skip probability (0.0-1.0) |
| `--skip_strategy` | `linear` | StepDrop strategy (see below) |
| `--output_dir` | `samples` | Output directory for images |
| `--save_grid` | `True` | Save samples as image grid |
| `--save_individual` | `False` | Save each sample individually |
| `--show` | `False` | Display samples (requires display) |

### Sampling Methods Comparison

| Method | NFE | Speed | Quality | Use Case |
|:-------|:----|:------|:--------|:---------|
| DDPM | 1000 | Slow | Best | Final production samples |
| DDIM (50) | 50 | Fast | Good | Quick iteration |
| DDIM (25) | 25 | Faster | Decent | Rapid prototyping |
| StepDrop | ~700 | Adaptive | Good | Balanced speed/quality |

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## StepDrop Samplers

StepDrop introduces stochastic step skipping for faster sampling with controllable quality tradeoffs.

### Available Skip Strategies

| Strategy | Formula | Description |
|:---------|:--------|:------------|
| `constant` | `p(t) = base_prob` | Fixed skip probability |
| `linear` | `p(t) = base_prob * 4t(1-t)` | Parabolic, peaks at middle |
| `cosine_sq` | `p(t) = base_prob * sin²(πt)` | Smooth cosine curve |
| `quadratic` | `p(t) = base_prob * 16t²(1-t)²` | Sharper peak in middle |
| `early_skip` | `p(t) = base_prob * t` | Skip more early (high noise) |
| `late_skip` | `p(t) = base_prob * (1-t)` | Skip more late (low noise) |
| `critical_preserve` | Variable | Low skip in [0.3, 0.7], high elsewhere |

### Usage Examples

```bash
# Linear (default) - balanced approach
python sample.py --checkpoint model.pt --method stepdrop \
  --skip_prob 0.3 --skip_strategy linear

# Quadratic - more aggressive middle skipping
python sample.py --checkpoint model.pt --method stepdrop \
  --skip_prob 0.5 --skip_strategy quadratic

# Critical preserve - protect important timesteps
python sample.py --checkpoint model.pt --method stepdrop \
  --skip_prob 0.4 --skip_strategy critical_preserve

# Adaptive - error-based dynamic skipping
python sample.py --checkpoint model.pt --method adaptive_stepdrop \
  --skip_prob 0.2
```

### Programmatic Usage

```python
from src.sampler import DDPMSampler, DDIMSampler, StepDropSampler, AdaptiveStepDropSampler

# DDPM (baseline)
sampler = DDPMSampler(num_timesteps=1000)
samples = sampler.sample(model, shape=(16, 3, 32, 32), device="cuda")

# DDIM (fast)
sampler = DDIMSampler(num_timesteps=1000, num_inference_steps=50, eta=0.0)
samples = sampler.sample(model, shape, device="cuda")

# StepDrop
sampler = StepDropSampler(num_timesteps=1000)
samples, stats = sampler.sample(
  model, shape, device="cuda",
  skip_prob=0.3,
  skip_strategy="linear"
)
print(f"Steps taken: {stats.steps_taken}, Skipped: {stats.steps_skipped}")

# Adaptive StepDrop
sampler = AdaptiveStepDropSampler(num_timesteps=1000)
samples, stats = sampler.sample(
  model, shape, device="cuda",
  base_skip_prob=0.2
)
```

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
│  ├── image3.png
│  └── image4.jpg
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

> Tip: CLI arguments override config file values.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## CLI Reference

### `src/train.py`

```bash
python train.py --help
```

### `src/sample.py`

```bash
python sample.py --help
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Examples

### Quick Test on MNIST

```bash
python train.py --epochs 5
python sample.py --checkpoint checkpoints/model.pt
```

### Train on CIFAR-10 with Larger Model

```bash
python train.py \
  --dataset cifar10 \
  --epochs 100 \
  --base_channels 128 \
  --batch_size 64
```

### Fast Sampling with DDIM

```bash
python sample.py \
  --checkpoint checkpoints/model.pt \
  --method ddim \
  --ddim_steps 25 \
  --n_samples 64
```

### StepDrop with Different Strategies

```bash
# Linear (default)
python sample.py --checkpoint model.pt --method stepdrop --skip_strategy linear

# Quadratic (more aggressive)
python sample.py --checkpoint model.pt --method stepdrop --skip_strategy quadratic --skip_prob 0.5

# Adaptive (error-based)
python sample.py --checkpoint model.pt --method adaptive_stepdrop
```

### Train on Your Own Photos

```bash
python train.py \
  --dataset custom \
  --custom_data_dir ~/my_photos \
  --img_size 128 \
  --channels 3 \
  --batch_size 32 \
  --epochs 200
```

### Resume Interrupted Training

```bash
python train.py --resume checkpoints/checkpoint_epoch_50.pt
```

### Generate Many Samples

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