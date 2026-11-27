# 3D Super-Resolution GAN for Protein Volumes

**BTP-3 | Bachelor Thesis Project**  
**IIIT Delhi | B.Tech Computer Science**

A research project focused on extending 2D GAN-based super-resolution techniques to 3D volumetric data, with an emphasis on protein-related volumes stored in MRC/MRCS formats.

---

## Overview

Super-resolution (SR) aims to reconstruct a high-resolution representation from a low-resolution input.

In this project, we explore the transition from conventional 2D image super-resolution to **3D volumetric super-resolution using Generative Adversarial Networks (GANs)**.

Unlike 2D images, 3D volumes contain spatial information along three dimensions. Our objective is to learn a reconstruction model that can enhance the quality and structural details of low-resolution 3D volumes while preserving important volumetric information.

The project has evolved through multiple experimental stages, including architecture modifications, data degradation experiments, training optimization, and evaluation of reconstructed volumes.

## Research Journey

### BTP-1: Understanding Generative Adversarial Networks

During BTP-1, we studied the fundamentals of GANs and explored several architectures:

- Vanilla GAN
- Deep Convolutional GAN (DCGAN)
- Conditional GAN (CGAN)
- Wasserstein GAN (WGAN)
- Pix2Pix
- CycleGAN

This phase established the foundation for understanding adversarial learning, image generation, and image-to-image translation.

### BTP-2: Super-Resolution and Style Generation

During BTP-2, our work moved toward advanced generative models:

- Super-Resolution GAN (SRGAN)
- StyleGAN

SRGAN introduced the concept of reconstructing high-resolution images from low-resolution inputs and became the foundation for our BTP-3 research.

### BTP-3: From 2D SRGAN to 3D SRGAN

In BTP-3, we began adapting the super-resolution pipeline from 2D images to 3D volumes.

The initial experiments involved general 3D image data. We subsequently shifted toward protein-related volumetric data, including MRC/MRCS files.

The main challenges included:

- Handling volumetric data instead of 2D images.
- Designing a suitable 3D generator architecture.
- Preparing paired high-resolution and low-resolution volumes.
- Maintaining spatial continuity across reconstructed volumes.
- Managing the computational cost of 3D training.
- Reducing overfitting and reconstruction artifacts.

---

## Project Pipeline

The overall workflow consists of the following stages:

```text
Protein / 3D Volumetric Data
            |
            v
      MRC / MRCS Loading
            |
            v
   Volume Preprocessing
            |
            v
   High-Resolution Volume
            |
            v
   HR-to-LR Degradation
            |
            v
 Low-Resolution Input Volume
            |
            v
      3D SRGAN Model
            |
      +-----+------+
      |            |
      v            v
   Generator   Discriminator
      |            |
      +-----+------+
            |
            v
    Training and Validation
            |
            v
    Trained SR Generator
            |
            v
   3D Super-Resolution Output
            |
            v
   Volumetric Evaluation
```

The pipeline is designed to learn a mapping from low-resolution volumetric inputs to reconstructed high-resolution volumes.

---

## Data Preparation and HR-to-LR Degradation

The project works with 3D volumetric data, including protein-related MRC/MRCS files.

The HR-to-LR degradation pipeline creates low-resolution inputs from high-resolution volumes for supervised training.

### Gaussian Noise

The current degradation experiments use **Gaussian noise with a noise level of 0.6**.

Gaussian noise is used to introduce controlled perturbations into the volume during the HR-to-LR preparation process.

The implementation also contains functions for other noise types. However, the current experiments primarily use Gaussian noise.

The exact interpretation of the noise parameter and its position in the degradation sequence should be verified from the implementation and configuration.

### Data Preparation Stages

1. Load the volumetric data.
2. Prepare and normalize the volume.
3. Generate the high-resolution representation.
4. Apply the selected degradation process.
5. Prepare the corresponding low-resolution input.
6. Use paired HR/LR volumes for model training and evaluation.

---

## Model Architecture

The project adapts a super-resolution GAN architecture for 3D volumetric reconstruction.

The GAN framework consists of two primary components:

### Generator

The generator learns to reconstruct a high-resolution volume from a low-resolution input.

Its objective is to produce a reconstructed volume that is structurally and quantitatively close to the target HR volume.

### Discriminator

The discriminator learns to distinguish real high-resolution volumes from generated super-resolution volumes.

The adversarial interaction between the generator and discriminator encourages the generator to produce more realistic reconstructions.

### Architecture Evolution

During development, the model architecture underwent several modifications.

| Stage | Architecture / Change |
|---|---|
| Initial implementation | SRResNet-16 |
| Later implementation | ResNet-50-based architecture |
| Kernel configuration | Experimented with changing 9×9×9 to 3×3×3 |

These modifications were part of the effort to improve reconstruction quality and training behavior.

The exact layer configuration, kernel placement, and parameter counts should be obtained from the model implementation and configuration files.

---

## Training and Evaluation

The project includes training, testing, and evaluation components.

Important repository components include:

- `training/config.py` — training configuration
- `training/ops.py` — training-related operations
- `training/srgan_model.py` — SRGAN model implementation
- `testing/test.py` — testing workflow
- `Evaluation/evaluate.py` — evaluation pipeline
- `Evaluation/checkpoint_evaluator.py` — checkpoint evaluation
- `Dataset/download_hr_mrc.py` — HR MRC data download utility
- `Dataset/split.py` — dataset splitting utility
- `Conversion/hr_to_lr.py` — HR-to-LR conversion

The repository also contains training history, validation summaries, evaluation summaries, and visualization outputs.

### Evaluation

The project evaluates reconstructed volumes using quantitative metrics and visual comparisons.

Evaluation artifacts include:

- Checkpoint metric comparisons
- Evaluation summaries
- Validation summaries
- Training and validation curves
- FSC-related plots
- Pipeline visualizations

The reported current performance is approximately **80%**, based on the project's current results.

The exact metric represented by this percentage should be specified alongside the evaluation protocol before making a formal scientific claim.

---

## Challenges and Improvements

### 1. Straight-Line Gaps in Reconstruction

During early experiments, generated outputs exhibited visible straight-line gaps or discontinuities.

These artifacts motivated further investigation of volumetric reconstruction, data preparation, and model behavior.

The objective of subsequent modifications was to improve continuity and reduce visible reconstruction artifacts.

### 2. Overfitting

The model showed signs of overfitting during training.

Overfitting occurs when a model learns the training data too closely and fails to generalize well to unseen volumes.

Training and validation monitoring, model selection, and architecture experimentation were used as part of the ongoing effort to improve generalization.

### 3. Architecture Modification

The transition from SRResNet-16 to a ResNet-50-based architecture and the kernel-size experiments were important stages in the model development process.

These experiments were intended to improve the model's ability to learn volumetric features and produce better reconstructions.

---

## Repository Structure

```text
BTP_3D/
│
├── Conversion/
│   └── hr_to_lr.py
│
├── Dataset/
│   ├── download_hr_mrc.py
│   └── split.py
│
├── Evaluation/
│   ├── evaluate.py
│   └── checkpoint_evaluator.py
│
├── testing/
│   └── test.py
│
├── training/
│   ├── config.py
│   ├── ops.py
│   └── srgan_model.py
│
├── model/
│   └── Model checkpoints and saved weights
│
├── Training and Evaluation Logs
│
└── README.md
```

The repository also contains evaluation plots, validation summaries, training histories, and other experiment artifacts.

---

## Current Status

The project has progressed from an initial 2D super-resolution approach toward a 3D GAN-based volumetric reconstruction pipeline.

Current development includes:

- 3D super-resolution model implementation
- Protein-related volumetric data processing
- HR-to-LR degradation using Gaussian noise
- Additional implemented noise functions
- Training and validation workflows
- Checkpoint-based evaluation
- Reconstruction and evaluation visualizations
- Architecture experimentation

The current reported performance is approximately 80%. Further work is required to document the exact metric, evaluation dataset, and reproducibility details.

---

## Future Work

Potential directions for further improvement include:

- Improving volumetric reconstruction continuity.
- Reducing overfitting.
- Experimenting with alternative degradation strategies.
- Evaluating additional noise types.
- Improving quantitative evaluation and reproducibility.
- Exploring more suitable 3D generator architectures.
- Increasing the robustness of reconstruction on unseen protein volumes.
- Documenting the exact evaluation metric and experimental protocol.

---

## Acknowledgements

This project is part of the Bachelor Thesis Project (BTP) undertaken at IIIT Delhi.

The work builds upon the concepts of GANs and super-resolution studied during BTP-1 and BTP-2 and extends them toward 3D volumetric reconstruction in BTP-3.

---

**Repository:** [BTP_3D](https://github.com/rahul-version/BTP_3D)