<a name="readme-top"></a>
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]
[![Buy Me a Coffee](https://img.shields.io/badge/Buy%20Me%20a%20Coffee-FFDD00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black)](https://www.buymeacoffee.com/wanghley)

<br />
<div align="center">
  <a href="https://github.com/wanghley/stepdrop-tiny-diffusion">
    <img src="docs/Figures/plot_residuals.png" alt="StepDrop Demo" width="280">
  </a>

  <h3 align="center">StepDrop</h3>

  <p align="center">
    Stochastic Step Skipping in Tiny Diffusion Models
    <br />
    <a href="https://github.com/wanghley/stepdrop-tiny-diffusion/blob/main/StepDrop_in_Stable_Diffusion_1.5.ipynb"><strong>Explore the Demo Notebook »</strong></a>
    <br />
  </p>
</div>

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a></li>
    <li><a href="#built-with">Built With</a></li>
    <li><a href="#getting-started">Getting Started</a></li>
    <li><a href="#quick-start">Quick Start</a></li>
    <li><a href="#pipeline-script">Pipeline Script</a></li>
    <li><a href="#training">Training</a></li>
    <li><a href="#sampling">Sampling</a></li>
    <li><a href="#evaluation--benchmarking">Evaluation & Benchmarking</a></li>
    <li><a href="#interpreting-metrics">Interpreting Metrics</a></li>
    <li><a href="#stepdrop-skip-strategies">StepDrop Skip Strategies</a></li>
    <li><a href="#visualization-utilities">Visualization Utilities</a></li>
    <li><a href="#project-structure">Project Structure</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

## About The Project

**StepDrop** is a novel sampling method designed to accelerate inference in diffusion models, particularly tiny ones. By introducing a stochastic step skipping technique, it significantly reduces the number of required sampling steps while maintaining high-quality image generation.

This repository contains the official implementation, experiments, and demo notebooks for the StepDrop project.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Built With

<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" style="vertical-align:top; margin:4px">
<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch" style="vertical-align:top; margin:4px">
<img src="https://img.shields.io/badge/Hugging_Face-FFD21E?style=for-the-badge&logo=hugging-face&logoColor=black" alt="Hugging Face" style="vertical-align:top; margin:4px">
<img src="https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white" alt="Jupyter" style="vertical-align:top; margin:4px">

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Getting Started

### Prerequisites

- Python 3.8+
- pip or conda
- CUDA-compatible GPU (recommended)

### Installation

1. Clone the repository
   ```sh
   git clone https://github.com/wanghley/stepdrop-tiny-diffusion.git
   cd stepdrop-tiny-diffusion
   ```

2. Create a virtual environment (recommended)
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies
   ```sh
   pip install -r requirements.txt
   ```

4. Verify installation
   ```sh
   python scripts/checklibs.py
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Quick Start

### Interactive Menu

The easiest way to get started is with the interactive quick start menu:

```bash
./scripts/quick_start.sh
```

This provides a menu-driven interface for common tasks like training, sampling, and benchmarking.

### One-Command Pipeline

Run the full pipeline (train → sample → evaluate) with a single command:

```bash
chmod +x scripts/pipeline.sh
./scripts/pipeline.sh --all --dataset cifar10 --epochs 10 --eval-samples 1000
```

### Quick Test

For a fast sanity check on MNIST:

```bash
./scripts/pipeline.sh --all --dataset mnist --epochs 5 --n-samples 16 --eval-samples 100
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Pipeline Script

The main automation tool is `scripts/pipeline.sh`. It orchestrates training, sampling, and evaluation.

### Usage

```bash
./scripts/pipeline.sh [OPTIONS]
```

### Pipeline Stages

| Flag | Description |
|:-----|:------------|
| `--train` | Run training stage |
| `--sample` | Run sampling stage |
| `--evaluate` | Run evaluation/benchmarking |
| `--all` | Run all stages (train → sample → evaluate) |
| `--clean` | Clean generated files |

### Common Options

| Option | Default | Description |
|:-------|:--------|:------------|
| `--dataset` | `cifar10` | Dataset: `mnist`, `cifar10`, `custom` |
| `--epochs` | `50` | Training epochs |
| `--batch-size` | `128` | Training batch size |
| `--base-channels` | `64` | U-Net base channels |
| `--checkpoint` | auto | Path to model checkpoint |
| `--n-samples` | `64` | Number of samples to generate |
| `--method` | `ddim` | Sampling method: `ddpm`, `ddim`, `stepdrop` |
| `--eval-samples` | `1000` | Samples for FID/IS evaluation |
| `--device` | `cuda` | Device: `cuda` or `cpu` |

### Examples

```bash
# Full CIFAR-10 training with evaluation
./scripts/pipeline.sh --all --dataset cifar10 --epochs 100 --base-channels 128 --eval-samples 5000

# Train only on MNIST
./scripts/pipeline.sh --train --dataset mnist --epochs 20

# Sample with DDIM from existing checkpoint
./scripts/pipeline.sh --sample --checkpoint checkpoints/model.pt --method ddim --ddim-steps 50 --n-samples 64

# Sample with StepDrop
./scripts/pipeline.sh --sample --checkpoint checkpoints/model.pt --method stepdrop --skip-prob 0.3 --skip-strategy linear

# Evaluate with full metrics
./scripts/pipeline.sh --evaluate --checkpoint checkpoints/model.pt --eval-samples 5000 --full-metrics

# Compare StepDrop strategies against DDIM baselines
./scripts/pipeline.sh --evaluate --checkpoint checkpoints/model.pt --compare-stepdrop --eval-samples 1000

# Dry run (show commands without executing)
./scripts/pipeline.sh --all --dataset mnist --epochs 5 --dry-run
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Training

### Direct Training Script

```bash
python src/train.py --dataset cifar10 --epochs 50 --batch_size 128
```

### Training Options

| Argument | Default | Description |
|:---------|:--------|:------------|
| `--dataset` | `mnist` | Dataset: `mnist`, `cifar10`, `custom` |
| `--custom_data_dir` | `None` | Path to custom images folder |
| `--img_size` | `28` | Image size |
| `--channels` | `1` | Number of image channels |
| `--batch_size` | `128` | Training batch size |
| `--epochs` | `20` | Number of epochs |
| `--lr` | `2e-4` | Learning rate |
| `--n_timesteps` | `1000` | Diffusion timesteps |
| `--schedule_type` | `cosine` | Noise schedule: `linear`, `cosine` |
| `--base_channels` | `64` | U-Net base channels |
| `--save_path` | `checkpoints/model.pt` | Model save path |
| `--resume` | `None` | Resume from checkpoint |

### Resume Training

```bash
python src/train.py --resume checkpoints/checkpoint_epoch_50.pt --epochs 100
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Sampling

### Direct Sampling Script

```bash
python src/sample.py --checkpoint checkpoints/model.pt --method ddim --ddim_steps 50 --n_samples 16
```

### Sampling Methods

| Method | Command | Description |
|:-------|:--------|:------------|
| **DDPM** | `--method ddpm` | Full 1000 steps, highest quality |
| **DDIM** | `--method ddim --ddim_steps 50` | Accelerated, deterministic |
| **StepDrop** | `--method stepdrop --skip_prob 0.3` | Stochastic step skipping |
| **Adaptive StepDrop** | `--method adaptive_stepdrop` | Error-based dynamic skipping |

### Sampling Options

| Argument | Default | Description |
|:---------|:--------|:------------|
| `--checkpoint` | required | Path to trained model |
| `--method` | `ddpm` | Sampling method |
| `--n_samples` | `16` | Number of samples |
| `--ddim_steps` | `50` | DDIM inference steps |
| `--ddim_eta` | `0.0` | DDIM stochasticity (0 = deterministic) |
| `--skip_prob` | `0.3` | StepDrop skip probability |
| `--skip_strategy` | `linear` | StepDrop strategy |
| `--output_dir` | `samples` | Output directory |
| `--save_grid` | `True` | Save as image grid |
| `--save_individual` | `False` | Save individual images |

### Examples

```bash
# DDPM (best quality, slow)
python src/sample.py --checkpoint checkpoints/model.pt --method ddpm --n_samples 16

# DDIM (fast)
python src/sample.py --checkpoint checkpoints/model.pt --method ddim --ddim_steps 25 --n_samples 64

# StepDrop with linear strategy
python src/sample.py --checkpoint checkpoints/model.pt --method stepdrop --skip_prob 0.3 --skip_strategy linear

# StepDrop with quadratic strategy (more aggressive)
python src/sample.py --checkpoint checkpoints/model.pt --method stepdrop --skip_prob 0.5 --skip_strategy quadratic

# Adaptive StepDrop
python src/sample.py --checkpoint checkpoints/model.pt --method adaptive_stepdrop
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Evaluation & Benchmarking

### Benchmark Script

Run comprehensive benchmarks comparing different sampling strategies:

```bash
# Quick test with dummy model
python scripts/benchmark_strategies.py --dummy --samples 10

# Full benchmark with trained model
python scripts/benchmark_strategies.py --checkpoint checkpoints/model.pt --samples 5000

# With full metrics (FID, KID, IS, Precision, Recall, LPIPS, SSIM, PSNR, Vendi)
python scripts/benchmark_strategies.py --checkpoint checkpoints/model.pt --samples 5000 --full-metrics
```

### Via Pipeline

```bash
# Basic evaluation
./scripts/pipeline.sh --evaluate --checkpoint checkpoints/model.pt --eval-samples 1000

# Full metrics
./scripts/pipeline.sh --evaluate --checkpoint checkpoints/model.pt --eval-samples 5000 --full-metrics

# Compare all StepDrop strategies vs DDIM
./scripts/pipeline.sh --evaluate --checkpoint checkpoints/model.pt --compare-stepdrop

# Evaluate only StepDrop variants
./scripts/pipeline.sh --evaluate --checkpoint checkpoints/model.pt --stepdrop-only

# Specific strategies only
./scripts/pipeline.sh --evaluate --checkpoint checkpoints/model.pt \
  --strategies "DDIM_50,StepDrop_Linear_0.3,StepDrop_Quadratic_0.3"
```

### Output

Results are saved to `results/<timestamp>/`:
- `report.json` - Full metrics data
- `report.csv` - Summary for Excel/Sheets
- `*.png` - Auto-generated plots (Pareto frontier, radar charts, etc.)
- `samples/` - Generated sample images per strategy

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Interpreting Metrics

| Metric | Full Name | Goal | Description |
|:-------|:----------|:-----|:------------|
| **FID** | Fréchet Inception Distance | 📉 Lower is better | Similarity to real dataset. <10: excellent, 10-30: good, >50: poor |
| **IS** | Inception Score | 📈 Higher is better | Clarity and diversity. CIFAR-10 real data ≈ 11.0 |
| **KID** | Kernel Inception Distance | 📉 Lower is better | Similar to FID, less biased for small samples |
| **Precision** | - | 📈 Higher is better | Quality: are generated images realistic? |
| **Recall** | - | 📈 Higher is better | Diversity: does the model cover the data distribution? |
| **LPIPS** | Perceptual Similarity | 📉 Lower is better | Perceptual distance (diversity among samples) |
| **Throughput** | Images/Second | 📈 Higher is better | Generation speed |
| **NFE** | Number of Function Evaluations | 📉 Lower is better | U-Net forward passes per image |

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## StepDrop Skip Strategies

### Probability-Based Strategies (`StepDropSampler`)

| Strategy | Formula | Description |
|:---------|:--------|:------------|
| `constant` | $P(t) = p$ | Fixed skip probability |
| `linear` | $P(t) = p \cdot 4t(1-t)$ | Parabolic peak at middle |
| `cosine_sq` | $P(t) = p \cdot \sin^2(\pi t)$ | Smooth cosine curve |
| `quadratic` | $P(t) = p \cdot 16t^2(1-t)^2$ | Sharper middle peak |
| `early_skip` | $P(t) = p \cdot t$ | Skip more at high noise |
| `late_skip` | $P(t) = p \cdot (1-t)$ | Skip more at low noise |
| `critical_preserve` | Variable | Protect [0.3, 0.7] interval |

### Adaptive Strategy (`AdaptiveStepDropSampler`)

Dynamically adjusts skipping based on reconstruction error:
- Low error → skip more aggressively
- High error → force denoising steps

### Target NFE Strategy (`TargetNFEStepDropSampler`)

Targets a specific step budget:
- `uniform` - Evenly spaced (like DDIM)
- `importance` - More steps at start/end
- `stochastic` - Random with boundary protection

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Visualization Utilities

### Generate Comparison Grid

```bash
python scripts/generate_grid.py
```
Output: `results/comparison_grid.png` - Side-by-side DDPM vs DDIM vs StepDrop

### Visualize Schedules

```bash
python scripts/plot_schedules.py --save_path results/schedules.png
```
Output: Probability curves and step sizes for different strategies

### Benchmark Plots

```bash
python scripts/plot_results.py --results results/2025-12-07_12-00-00/
```
Output: Pareto frontiers, radar charts, metric comparisons

### Denoising Evolution

```bash
python scripts/plot_denoising_evolution.py
```
Output: `results/plot_denoising_evolution.png` - Film strip showing denoising progression

### Efficiency Plots

```bash
python scripts/plot_efficiency.py --results results/
```
Output: FLOPs/Memory analysis

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Project Structure

```
stepdrop-tiny-diffusion/
├── src/
│   ├── config.py          # Configuration management
│   ├── dataset.py         # Data loading (MNIST, CIFAR-10, custom)
│   ├── modules.py         # U-Net architecture
│   ├── scheduler.py       # Noise schedules
│   ├── train.py           # Training script
│   ├── sample.py          # Sampling script
│   ├── sampler/           # Sampler implementations
│   │   ├── DDPM.py
│   │   ├── DDIM.py
│   │   ├── StepDrop.py
│   │   └── AdaptiveStepDrop.py
│   └── eval/              # Evaluation metrics
│       └── metrics_utils.py
├── scripts/
│   ├── pipeline.sh        # Main automation script
│   ├── quick_start.sh     # Interactive menu
│   ├── benchmark_strategies.py
│   ├── plot_results.py
│   ├── plot_schedules.py
│   ├── plot_denoising_evolution.py
│   └── generate_grid.py
├── notebooks/             # Jupyter notebooks
├── checkpoints/           # Saved models
├── samples/               # Generated samples
├── results/               # Benchmark results
└── docs/                  # Documentation
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## HPC / SLURM Support

For cluster environments:

```bash
# Submit job to SLURM
sbatch scripts/run_pipeline.slurm

# With custom arguments
sbatch scripts/run_pipeline.slurm --train --dataset cifar10 --epochs 100
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Roadmap

- [x] Core StepDrop sampler implementation
- [x] Pipeline automation script
- [x] Comprehensive benchmarking suite
- [x] Multiple skip strategies
- [x] Example notebook for Tiny Diffusion
- [x] Example notebook for Stable Diffusion 1.5
- [ ] Package as pip-installable library
- [ ] Integration with HuggingFace Diffusers
- [ ] Support for more diffusion schedulers

See [open issues](https://github.com/wanghley/stepdrop-tiny-diffusion/issues) for proposed features and known issues.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Contributing

Contributions are welcome!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## License

Distributed under the MIT License. See `LICENSE` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Contact

Wanghley Soares Martins - [@wanghley](https://instagram.com/wanghley) - me@wanghley.com

Project Link: [https://github.com/wanghley/stepdrop-tiny-diffusion](https://github.com/wanghley/stepdrop-tiny-diffusion)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Acknowledgments

- [Hugging Face Diffusers](https://github.com/huggingface/diffusers)
- [Choose an Open Source License](https://choosealicense.com)
- [Img Shields](https://shields.io)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->

[contributors-shield]: https://img.shields.io/github/contributors/wanghley/stepdrop-tiny-diffusion?style=for-the-badge
[contributors-url]: https://github.com/wanghley/stepdrop-tiny-diffusion/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/wanghley/stepdrop-tiny-diffusion.svg?style=for-the-badge
[forks-url]: https://github.com/wanghley/stepdrop-tiny-diffusion/network/members
[stars-shield]: https://img.shields.io/github/stars/wanghley/stepdrop-tiny-diffusion.svg?style=for-the-badge
[stars-url]: https://github.com/wanghley/stepdrop-tiny-diffusion/stargazers
[issues-shield]: https://img.shields.io/github/issues/wanghley/stepdrop-tiny-diffusion.svg?style=for-the-badge
[issues-url]: https://github.com/wanghley/stepdrop-tiny-diffusion/issues
[license-shield]: https://img.shields.io/github/license/wanghley/stepdrop-tiny-diffusion.svg?style=for-the-badge
[license-url]: https://github.com/wanghley/stepdrop-tiny-diffusion/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/wanghley
[product-screenshot]: images/screenshot.png
