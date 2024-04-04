# Generative AI: Denoising Autoencoder Final Report

**Authors:** Zahra Rahmani, Adi Ghosh  
**Department:** Computer Science, Case Western Reserve University  
**Contact:**  
- Zahra Rahmani: zxr81@case.edu  
- Adi Ghosh: axg1328@case.edu  

## Abstract
The primary aim of this project is to identify the most suitable Denoising Autoencoder (DAE) architecture for handling specific noise profiles in corrupted images. By employing a rigorous methodology involving grid search and systematic exploration of different combinations of hyperparameters and architectures, we aim to enhance the quality and usability of noisy images across various applications.

## Introduction
In this study, we aim to discern the effectiveness of Variational Autoencoders (VAEs) and Denoising Autoencoders (DAEs) in image denoising tasks across various noise profiles. Image denoising is crucial in numerous applications, from medical imaging to digital photography. This report delves into the methodology, results, and implications of our findings.

### Denoising potential of generative models
Our goal is to identify the best deep learning architectures for cleaning up noisy images, specifically exploring VAEs and DAEs across different types of noise. This study is crucial for improving the quality of images that are corrupted by noise, making them more useful in a variety of applications.

### The challenge of image denoising
The complexity of our problem lies in the variety of ways images can become corrupted by noise during acquisition and transmission. Addressing this challenge is crucial, as developing effective denoising models plays a significant role in fields where high-quality image data is essential for precise analysis and decision-making.

### Objectives of the denoising exploration
In our exploration of image denoising, we set several key objectives, including methodically evaluating different configurations of VAEs and DAEs across a range of noise profiles, identifying the most effective autoencoder architecture for each noise type, and optimizing resource usage for practical deployment.

## Methodology
### Data source and noise types
We utilized the MNIST and Fashion-MNIST datasets as foundational sets of clean images and introduced a variety of noise profiles into these datasets to challenge our models and simulate real-world scenarios.

### Architectural choices in denoising
We experimented with varying depths of encoder and decoder layers, different activation functions, and training durations to assess the efficiency of image reconstruction and noise reduction.

### Loss function
We selected the Kullback-Leibler (KL) Divergence as our primary loss function due to its effectiveness in the context of Denoising Autoencoders (DAEs) and the denoising process.

## Results and Analysis
### Results
From the grid search, we identified optimal architectures for different noise profiles, with varying numbers of encoder layers, decoder layers, and activation functions.

### Novelty
Our project fills a significant gap in academic research regarding the identification of the most suitable architecture for each specific type of noise in image processing.

### Challenges and Limitations
Technical challenges such as automatic shutdown of jobs in the High-Performance Computing (HPC) environment and the complexity of finding the best architecture were encountered during the project.

## Conclusion and Future Direction
Our experiments have yielded promising data indicating optimal combinations of variables for configuring DAE architectures. Looking ahead, we plan to broaden the scope of our research to include a wider array of image datasets and explore the integration of more advanced neural network techniques.

## References
- Bengio, Yoshua, et al. "Generalized denoising auto-encoders as generative models." Advances in neural information processing systems 26 (2013).
- Gondara, L. "Medical Image Denoising Using Convolutional Denoising Autoencoders," 2016 IEEE 16th International Conference on Data Mining Workshops (ICDMW), Barcelona, Spain, 2016.
