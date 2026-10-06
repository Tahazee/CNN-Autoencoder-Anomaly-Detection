# Industrial Sense AI

### Unsupervised Visual Inspection & Anomaly Detection

![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat\&logo=pytorch\&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-UI-orange)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

An end-to-end **unsupervised visual inspection and anomaly detection framework** built with **PyTorch** and **Gradio**.

Industrial Sense AI learns the visual characteristics of defect-free industrial components and identifies deviations from the learned representation. The system is designed for high-resolution industrial inspection scenarios and can be evaluated on datasets such as **MVTec AD**.

The framework provides both **image-level anomaly detection** and **pixel-level defect localization**, allowing users to identify not only whether an object is defective, but also where the potential defect occurs.

---

## Key Features

### 1. Convolutional Autoencoder

A custom convolutional autoencoder is used to learn representations of defect-free samples.

* 4-stage convolutional encoder
* Compressed **128-dimensional bottleneck representation**
* 4-stage transposed-convolution decoder
* Reconstruction-based anomaly detection
* Trained primarily on defect-free images

The model learns to reconstruct normal samples accurately. Defective regions are expected to produce higher reconstruction errors.

---

### 2. Unsupervised Anomaly Detection

The system follows a reconstruction-based unsupervised learning approach.

During training, the model is exposed to **defect-free images** rather than requiring pixel-level defect annotations.

At inference time:

1. An inspection image is passed through the autoencoder.
2. The model reconstructs the image.
3. The original and reconstructed images are compared.
4. Reconstruction error is used as the anomaly signal.
5. Regions with elevated reconstruction error are highlighted as potential defects.

This enables anomaly detection without requiring manually labelled defect masks during training.

---

### 3. Pixel-Level Anomaly Localization

The framework generates a pixel-level reconstruction error map using an **L1 distance** between the input and reconstructed images.

The error map is then:

* Smoothed using Gaussian filtering
* Normalized for visualization
* Converted into a colorized anomaly heatmap
* Overlaid onto the original inspection image

This provides visual localization of suspicious regions rather than producing only a binary defective/normal prediction.

---

### 4. ECDF Error Diagnostics

The project includes **Empirical Cumulative Distribution Function (ECDF)** analysis for investigating reconstruction-error distributions.

Diagnostic plots can be used to compare:

* Normal samples
* Defective samples
* Pixel-level reconstruction errors
* Different anomaly decision thresholds

These diagnostics help analyze the separation between normal and anomalous reconstruction-error distributions.

---

### 5. Interactive Gradio Interface

A standalone Gradio application provides an interactive visual inspection interface.

Users can:

* Upload inspection images
* Select available dataset samples
* Run anomaly detection
* Adjust anomaly thresholds
* View reconstruction results
* Inspect pixel-level anomaly heatmaps
* Analyze anomaly scores visually

The interface is intended to make the model's predictions easier to inspect and interpret.

---

## System Pipeline

```text
             Input Inspection Image
                       │
                       ▼
              Image Preprocessing
                       │
                       ▼
             Convolutional Encoder
                       │
                       ▼
              128-D Bottleneck
                       │
                       ▼
             Transposed-Conv Decoder
                       │
                       ▼
              Reconstructed Image
                       │
              ┌────────┴────────┐
              ▼                 ▼
       Reconstruction       L1 Error Map
          Error Score              │
              │                    ▼
              │              Gaussian Smoothing
              │                    │
              │                    ▼
              │              Anomaly Heatmap
              │                    │
              └──────────┬─────────┘
                         ▼
                 Inspection Result
```

---

## Project Structure

```text
industrial-sense-ai/
│
├── code/
│   ├── datascience_project.py
│   │   └── Core model training and anomaly detection pipeline
│   │
│   ├── datascience_project_collab.ipynb
│   │   └── Google Colab experimentation and training notebook
│   │
│   └── debug.py
│       └── Diagnostic and testing utilities
│
├── app.py
│   └── Gradio-based visual inspection application
│
├── requirements.txt
│   └── Python dependencies
│
├── .gitignore
│   └── Excludes model weights, datasets and other large/generated files
│
└── README.md
    └── Project documentation
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/industrial-sense-ai.git
cd industrial-sense-ai
```

### 2. Create a Virtual Environment

#### Linux / macOS

```bash
python -m venv venv
source venv/bin/activate
```

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Dataset

The framework can be evaluated using industrial anomaly-detection datasets such as **MVTec AD**.

The training pipeline is designed around the unsupervised setting:

```text
Training:
Normal / defect-free images
            │
            ▼
       Autoencoder
            │
            ▼
    Learned normal representation


Inference:
Normal OR defective image
            │
            ▼
       Autoencoder
            │
            ▼
    Reconstruction Error
            │
            ▼
   Anomaly Score + Heatmap
```

The model should be trained using defect-free samples so that deviations from the learned normal appearance generate higher reconstruction errors.

---

## Model Training

To train the model using the provided training script:

```bash
python code/datascience_project.py
```

For GPU-accelerated experimentation, the Google Colab notebook can also be used:

```text
code/datascience_project_collab.ipynb
```

After training, save the resulting model checkpoint as:

```text
autoencoder_mvtec.pth
```

in the project root directory.

---

## Running the Application

Start the Gradio interface:

```bash
python app.py
```

The application will provide a local URL similar to:

```text
http://127.0.0.1:7860
```

Open the URL in a browser to access the visual inspection interface.

### Model Checkpoint

Before launching the application, ensure the trained model checkpoint is available:

```text
industrial-sense-ai/
├── app.py
├── autoencoder_mvtec.pth
└── ...
```

---

## Anomaly Detection

The
