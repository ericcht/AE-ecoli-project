# AE E. Coli Autoencoder Project

This project implements an **Autoencoder (AE)** model to analyze and compress **E. Coli** data. The Autoencoder aims to learn an efficient representation (encoding) of the input data, enabling applications such as data compression, anomaly detection, and dimensionality reduction. 


## Project Overview
The goal of this project is to implement an Autoencoder model that can:
- **Compress the input data** of E. Coli features into a lower-dimensional representation.
- **Reconstruct the data** from the compressed encoding to evaluate how well the Autoencoder can recover the original features.
- Explore the use of Autoencoders for **anomaly detection** and **feature extraction** in biological data.

Key features:
- **Data Preprocessing**: Encodes and normalizes the E. Coli dataset.
- **Autoencoder Architecture**: Defines the encoder and decoder layers of the Autoencoder model.
- **Model Training**: Implements training procedures to minimize the reconstruction error.
- **Evaluation**: Uses loss plots and reconstruction error to evaluate model performance.

## Installation
To get started with the project, clone the repository and install the required dependencies.

### Clone the repository
```bash
git clone https://github.com/ericcht/AE-ecoli-project.git
cd AE-ecoli-project
