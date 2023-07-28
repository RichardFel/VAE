# **Exploring Unsupervised Feature Extraction of IMU-Based Gait Data in Stroke Rehabilitation using a Variational Autoencoder**

## Introduction

To gain a more comprehensive understanding of gait recovery, monitor progress and tailor interventions, measuring the way people walk is crucial [1,2,3]. Inertial Measurement Units (IMUs) are small and portable sensors that enable objective and continuous measurements of the way people walk. However, IMU data needs to be processed to extract relevant information before it can be used in research and clinical practice.

This study explored a data-driven approach of processing IMU data using Variational AutoEncoder (VAE) [4]. A VAE is a generative model that employs deep learning techniques to learn a compact, low-dimensional representation of data.

## Variational AutoEncoder

The VAE comprises two main components: an encoder and a decoder. The encoder maps the input-data to a lower-dimensional representation, known as the latent layer, by encoding it into a mean and variance vector. This vector is then used to generate a sample from a probability distribution that models the latent layer. The decoder takes this sample as input and generates a reconstructed output that is similar to the original input data. The difference between the original input and the reconstructed output is measured using a loss function. The VAE aims to minimize this loss function, which encourages the encoder to learn a good representation of the input data in the latent layer.

The input and output of the VAE consisted of a 512X6 epoch. The encoder and decoder both comprised three convolutional layers. The latent layer contained 12 latent variables. Figure 1 provides a simplified visual representation of the VAE architecture.

<a href="https://imgbb.com/"><img src="https://i.ibb.co/mykQWsb/Picture-1.png" alt="Picture-1" border="0"></a>
Figure 1: Variational autoencoder (VAE) used in this study. The input was a 512X6 epoch of an IMU-based gait measurement. The encoder (green) and the decoder (blue) consisted of 3 mirrored convolutional layers with a size of 256, 128 and 64 nodes. These layers were configured with 32, 64, and 128 filters, respectively, and employed a kernel size of 3. The activation function used throughout the model was tanh. The latent layer contained 12 normally distributed latent features. The model was trained by comparing the input to the reconstructed output. A tanh activation function was used in the convolutional layers. An Adam optimizer with a learning rate of 0.001 was used.

## References

[1] Sung Shin, Robert Lee, Patrick Spicer, and James Sulzer. Does kinematic gait quality improve with functional gait recovery? a longitudinal pilot study on early post-stroke individuals. Journal of Biomechanics, 105:109761, 03 2020.

[2] Elizabeth Wonsetler and Mark Bowden. A systematic review of mechanisms of gait speed change post-stroke. part 1: spatiotemporal parameters and asymmetry ratios. Topics in Stroke Rehabilitation, 24:1–12, 02 2017.

[3] Michiel Punt, Sjoerd Bruijn, Kim van Schooten, Mirjam Pijnappels, Ingrid Port, Harriet Wittink, and Jaap Van Dieen. Characteristics of daily life gait in fall and non fall-prone stroke survivors and controls. Journal of NeuroEngineering and Rehabilitation, 13, 07 2016.

[4] Diederik Kingma and Max Welling. An Introduction to Variational Autoencoders. 01 2019.
