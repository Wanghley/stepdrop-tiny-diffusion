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

### Prerequisites

You will need Python 3.8+ and pip installed on your system.
* [Python](https://www.python.org/)
* [pip](https://pypi.org/project/pip/)

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/wanghley/stepdrop-tiny-diffusion.git
    ```

2.  Navigate to the project directory
    ```sh
    cd stepdrop-tiny-diffusion
    ```
3.  (Recommended) Create a virtual environment
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
4.  Install required packages
    ```sh
    pip install -r requirements.txt
    ```

<p align="right">(<a href="#readme-top">back to top<a>)<p\>

## Usage

The primary usage and examples are demonstrated in the included Jupyter Notebooks.

  * **`StepDrop_in_Stable_Diffusion_1.5.ipynb`**: Demonstrates how to apply StepDrop to the Stable Diffusion 1.5 model.
  * **`StepDrop_in_Tiny_Diffusion.ipynb`**: Shows the core StepDrop implementation with a smaller, tiny diffusion model.

Open and run these notebooks in a Jupyter environment to see how StepDrop is implemented and test its performance.

```bash
jupyter notebook
```

<p align="right"\>(<a href="#readme-top"\>back to top\<a\>)\<p\>

## Roadmap

  - [x] Core StepDrop sampler implementation
  - [x] Example notebook for Tiny Diffusion
  - [x] Example notebook for Stable Diffusion 1.5
  - [ ] Package the sampler as a pip-installable library
  - [ ] Add support for more diffusion schedulers
  - [ ] Integrate directly into the `diffusers` library

See the [open issues](https://www.google.com/search?q=https://github.com/wanghley/stepdrop-tiny-diffusion/issues) for a full list of proposed features and known issues.

<p align="right"\>(<a href="#readme-top"\>back to top<a\>)\<p\>

## Contributing

Contributions are welcome\! If you have suggestions, improvements, or bug fixes, feel free to open an issue or submit a pull request.

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

<p align="right">(<a href="#readme-top">back to top<a\>)<p\>

## License

Distributed under the MIT License. See `LICENSE` for more information.

<p align="right"\>(<a href="#readme-top"\>back to top<a\>)\<p\>

## Contact

Wanghley Soares Martins - [@wanghley](https://instagram.com/wanghley) - me@wanghley.com

Project Link: [https://github.com/wanghley/stepdrop-tiny-diffusion](https://github.com/wanghley/stepdrop-tiny-diffusion)

<p align="right">(<a href="#readme-top">back to top\<a\>)\<p\>

## Acknowledgments

  * [Hugging Face Diffusers](https://github.com/huggingface/diffusers)
  * [Choose an Open Source License](https://choosealicense.com)
  * [Img Shields](https://shields.io)

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
