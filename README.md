
---

# Pix2Pix Image-to-Image Translation

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

A PyTorch implementation of **pix2pix**, a conditional Generative Adversarial Network (cGAN) for general-purpose image-to-image translation. This project recreates the seminal work "[Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004)" by Isola et al. (2018).

![pix2pix Examples](https://phillipi.github.io/pix2pix/images/teaser_v3.png)
*Example of image-to-image translation tasks (Source: Original paper)*

## Project Overview

Pix2pix is a framework that learns a mapping from input images to output images using paired data. It has been successfully applied to various tasks, including:

- **Semantic Segmentation** 🖼️→🏷️
- **Map to Aerial Photo** 🗺️→🛰️
- **Black & White to Color** ⚫⚪→🌈
- **Edges to Photo** ✏️→📷
- **Day to Night** ☀️→🌙
- **Sketch to Portrait** 🎨→👨‍🎨

This implementation provides a clean, well-documented recreation of the pix2pix model using PyTorch, making it accessible for learning and experimentation.

## 🛠️ Installation & Setup

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for training)
- PyTorch

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/Hemanth2332/pix2pix-recreation.git
cd pix2pix-recreation
```

2. **Create a virtual environment (recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**

```bash
pip install torch torchvision torchaudio matplotlib pillow numpy opencv-python tqdm tensorboard
```
*It is recommended to install gpu compatible torch version*

## 🚀 Usage

### 1. Data Preparation

The model requires paired datasets where each input image has a corresponding target output image.

**Dataset Structure:**
```
dataset/
├── train/
│   ├── input/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── target/
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ...
└── val/
    ├── input/
    └── target/
```

**Popular datasets for pix2pix:**
- [Facades](https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/facades.tar.gz)
- [Maps](https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/maps.tar.gz)
- [Edges to Shoes/Handbags](https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/edges2shoes.tar.gz)


**Key Training Parameters:**
- `dataset_path`: Path to your dataset
- `epochs`: Number of training epochs (default: 200)
- `batch_size`: Batch size (default: 1)
- `lr`: Learning rate (default: 0.0002)
- `lambda_l1`: L1 loss weight (default: 100)


## 🏗️ Model Architecture

### Generator (U-Net)
- **Type:** Encoder-Decoder with skip connections
- **Input:** Source image (e.g., edges, semantic map)
- **Output:** Translated image (e.g., photo, color image)
- **Skip Connections:** Preserve low-level information between encoder and decoder

### Discriminator (PatchGAN)
- **Type:** Convolutional classifier
- **Input:** Concatenated source and target images OR source and generated images
- **Output:** Patch of probabilities representing real/fake classification per image patch
- **Receptive Field:** 70×70 patches (original paper)

## ⚙️ Training Details

### Loss Function
The model combines two loss functions:

```
Total Loss = Adversarial Loss + λ × L1 Loss
```

- **Adversarial Loss:** Standard GAN loss from the discriminator
- **L1 Loss:** Encourages pixel-wise similarity between generated and target images
- **λ:** Weight parameter (typically 100)

### Training Strategy
- **Optimizer:** Adam (β₁=0.5, β₂=0.999)
- **Learning Rate:** 0.0002 (constant for first 100 epochs, then linear decay)
- **Batch Normalization:** Used in both generator and discriminator
- **Instance Normalization:** Alternative for better stability

## 📊 Results & Evaluation

The model saves generated samples at regular intervals during training.

### Qualitative Evaluation
![result](result/result2.png)

## 🎯 Key Features

- ✅ **U-Net Generator** with skip connections
- ✅ **PatchGAN Discriminator** for high-frequency structure
- ✅ **Conditional GAN training** with paired data
- ✅ **L1 loss** for pixel-wise similarity
- ✅ **Modular code structure** for easy experimentation
- ✅ **Training monitoring** with sample generation

##  Future Enhancements

- [ ] Multi-GPU training support
- [ ] TensorBoard integration for better visualization
- [ ] Pre-trained models for common datasets
- [ ] Web demo with Gradio/Streamlit
- [ ] Support for additional loss functions (e.g., fourier loss)
- [ ] CycleGAN extension for unpaired image translation

## 📚 References

- [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004)
- [Official PyTorch Implementation](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)

---

**⭐ If you find this project useful, please consider giving it a star on GitHub!**

---

