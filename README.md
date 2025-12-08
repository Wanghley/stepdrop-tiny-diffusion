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
    <img src="image.gif" alt="StepDrop Demo" width="280">
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
    <li><a href="#usage">Usage</a></li>
    <li><a href="#pipeline-script">Pipeline Script</a></li>
    <li><a href="#evaluation--benchmarking">Evaluation & Benchmarking</a></li>
    <li><a href="#stepdrop-skip-strategies">StepDrop Skip Strategies</a></li>
    <li><a href="#developer-guide">Developer Guide</a></li>
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

To get a local copy up and running, follow these simple steps.

> Developer Guide: For module APIs and sampler details, see `src/README.md`.

### Prerequisites

You will need Python 3.8+ and pip installed on your system.

- [Python](https://www.python.org/)
- [pip](https://pypi.org/project/pip/)

### Installation

1. Clone the repo

   ```sh
   git clone https://github.com/wanghley/stepdrop-tiny-diffusion.git
   ```

2. Navigate to the project directory
   ```sh
   cd stepdrop-tiny-diffusion
   ```
3. (Recommended) Create a virtual environment
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```
4. Install required packages
   ```sh
   pip install -r requirements.txt
   ```

  ### Try It

  Jump straight into the pipeline and benchmarks:

  ```bash
  # Pipeline (train → sample → evaluate)
  chmod +x scripts/pipeline.sh
  ./scripts/pipeline.sh --all --dataset cifar10 --epochs 10 --eval-samples 1000

  # Benchmark (FID/IS/throughput)
  python scripts/benchmark_strategies.py --checkpoint checkpoints/model.pt --samples 1000
  ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Interpreting Metrics

When running `scripts/benchmark_strategies.py`, you will generate a `report.json`. Here is how to read the numbers:

| Metric         | Full Name                  | Goal                    | Description                                                                                                                                                            |
| :------------- | :------------------------- | :---------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **FID**        | Fréchet Inception Distance | **📉 Lower is better**  | Measures how similar the generated images are to the real dataset. <br>• **< 10**: State-of-the-art <br>• **10-30**: High quality <br>• **> 50**: Noticeable artifacts |
| **IS**         | Inception Score            | **📈 Higher is better** | Measures "clarity" (does it look like an object?) and "diversity" (are there many types of objects?). <br>• For CIFAR-10, real data has IS ≈ 11.0.                     |
| **Throughput** | Images Per Second          | **📈 Higher is better** | Raw generation speed. Higher throughput means lower latency.                                                                                                           |
| **MACs/FLOPs** | Multiply-Accumulates       | **📉 Lower is better**  | The theoretical computational cost. StepDrop aims to reduce this by skipping steps.                                                                                    |
| **Avg Steps**  | Average Steps Taken        | **📉 Lower is better**  | The average number of U-Net evaluations per image. <br>• **DDPM**: Always 1000 <br>• **DDIM**: Fixed (e.g., 50) <br>• **StepDrop**: Variable (e.g., ~700)              |

## Usage

### Command Line Interface (CLI)

We provide a powerful benchmarking harness to evaluate model speed and quality (FID/IS) from the terminal.

**Run Benchmark:**

```bash
# Dry run with dummy model (verify setup)
python scripts/benchmark_strategies.py --dummy --samples 10

# Full evaluation with trained model
python scripts/benchmark_strategies.py --checkpoint checkpoints/model.pt --samples 5000
```

- **Result**: Generates a `results/YYYY-MM-DD.../` folder with a `report.json` and sample images.
- **Help**: Run `python scripts/benchmark_strategies.py --help` for full options.

<p align="right"\>(<a href="#readme-top"\>back to top\<a\>)\<p\>

## Pipeline Script

The main automation entrypoint is `scripts/pipeline.sh`. It supports training, sampling, and comprehensive evaluation.

- Show help
```bash
chmod +x scripts/pipeline.sh
./scripts/pipeline.sh --help
```

- Full pipeline (train → sample → evaluate)
```bash
./scripts/pipeline.sh --all --dataset cifar10 --epochs 50 --eval-samples 1000
```

- Train only
```bash
./scripts/pipeline.sh --train --dataset mnist --epochs 20
```

- Sample only
```bash
# DDPM
./scripts/pipeline.sh --sample --checkpoint checkpoints/model.pt --method ddpm --n-samples 64
# DDIM (fast)
./scripts/pipeline.sh --sample --checkpoint checkpoints/model.pt --method ddim --ddim-steps 50 --n-samples 64
# StepDrop
./scripts/pipeline.sh --sample --checkpoint checkpoints/model.pt --method stepdrop \
  --skip-prob 0.3 --skip-strategy linear --n-samples 64
```

## Evaluation & Benchmarking

Run comprehensive benchmarks (FID, IS, throughput by default; add full metrics for KID, Precision/Recall, LPIPS, SSIM, PSNR, Vendi, etc.).

```bash
# Basic benchmark
./scripts/pipeline.sh --evaluate --checkpoint checkpoints/model.pt --eval-samples 1000

# Full metrics
./scripts/pipeline.sh --evaluate --checkpoint checkpoints/model.pt --eval-samples 5000 --full-metrics

# Compare all StepDrop strategies against DDIM baselines
./scripts/pipeline.sh --evaluate --checkpoint checkpoints/model.pt --compare-stepdrop --eval-samples 1000

# Evaluate only StepDrop variants
./scripts/pipeline.sh --evaluate --checkpoint checkpoints/model.pt --stepdrop-only --eval-samples 1000

# Specific strategies only
./scripts/pipeline.sh --evaluate --checkpoint checkpoints/model.pt \
  --strategies "DDIM_50,StepDrop_Linear_0.3,StepDrop_Quadratic_0.3" --eval-samples 1000
```

Results are saved under `results/<timestamp>/` with `report.json`, `report.csv`, and per-strategy sample images.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## StepDrop Skip Strategies

StepDrop introduces stochastic step skipping with configurable schedules:

- Linear: parabolic peak at middle (`4t(1-t)`)
- Cosine²: smooth `sin²(πt)` curve
- Quadratic: sharper middle peak (`16t²(1-t)²`)
- Constant: fixed probability throughout
- Early/Late Skip: bias skipping to early (high noise) or late (low noise) steps
- Critical Preserve: low skipping in the critical middle region

Examples:
```bash
./scripts/pipeline.sh --sample --method stepdrop --skip-strategy linear --skip-prob 0.3
./scripts/pipeline.sh --sample --method stepdrop --skip-strategy cosine_sq --skip-prob 0.3
./scripts/pipeline.sh --sample --method stepdrop --skip-strategy quadratic --skip-prob 0.5
```

To compare schedules:
```bash
./scripts/pipeline.sh --evaluate --compare-stepdrop --eval-samples 1000
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Developer Guide

For module-level details, sampler APIs, and CLI options, see the developer-focused docs:

- `src/README.md`: Core modules, `src/sampler/` (DDPM, DDIM, StepDrop, Adaptive), evaluation utilities, and examples.

Quick pointers:

- Try the pipeline: 
```bash
chmod +x scripts/pipeline.sh
./scripts/pipeline.sh --all --dataset cifar10 --epochs 10 --eval-samples 1000
```
- Run the benchmark directly:
```bash
python scripts/benchmark_strategies.py --checkpoint checkpoints/model.pt --samples 1000
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Roadmap

- [x] Core StepDrop sampler implementation
- [x] Example notebook for Tiny Diffusion
- [x] Example notebook for Stable Diffusion 1.5
- [ ] Package the sampler as a pip-installable library
- [ ] Add support for more diffusion schedulers
- [ ] Integrate directly into the `diffusers` library

See the [open issues](https://www.google.com/search?q=https://github.com/wanghley/stepdrop-tiny-diffusion/issues) for a full list of proposed features and known issues.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Contributing

Contributions are welcome\! If you have suggestions, improvements, or bug fixes, feel free to open an issue or submit a pull request.

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

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

<p align="right">(<a href="#readme-top">back to top<a\>)<p\>

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
[license-url]: https://github.com/wanghley/stepdrop-tiny-diffusionblob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/wanghley
[product-screenshot]: images/screenshot.png
