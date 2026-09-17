# Industrial Sense AI: Unsupervised Visual Inspection & Anomaly Detection

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Gradio](https://img.shields.io/badge/Gradio-UI-orange)](https://gradio.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end unsupervised visual inspection framework built with PyTorch and Gradio. This system utilizes a Convolutional Autoencoder with a compressed bottleneck vector to detect, locate, and measure manufacturing defects on high-resolution industrial components (e.g., MVTec AD dataset).

---

## Key Features

- **Convolutional Autoencoder Architecture**: Features 4 convolutional encoder stages down to a 128-dimensional bottleneck, paired with 4 transpose-convolutional decoder stages.
- **Unsupervised Anomaly Detection**: Learns standard feature representations exclusively from defect-free training images. Anomalous regions produce elevated reconstruction errors.
- **Pixel-Level Heatmap Overlay**: Computes L1 reconstruction distance map, applies Gaussian smoothing, and overlays colorized heatmaps (`JET` colormap) on original input images.
- **ECDF Error Diagnostics**: Generates Empirical Cumulative Distribution Function (ECDF) curves and diagnostic charts comparing normal vs. defective pixel error distributions.
- **Interactive Gradio Interface**: Standalone web UI allowing users to upload inspection images, select dataset samples, adjust anomaly decision thresholds dynamically, and inspect visual heatmaps.

---

## Project Structure

```text
├── code/
│   ├── datascience_project.py        # Core PyTorch Autoencoder & Anomalib training script
│   ├── datascience_project_collab.ipynb # Google Colab experiment notebook
│   └── debug.py                       # Diagnostic utilities & testing helpers
├── app.py                             # Gradio web application for visual inspection UI
├── requirements.txt                   # Python environment dependencies
└── .gitignore                         # Configured to exclude heavy binaries & weights
```

---

## Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/<your-username>/anomaly-detection.git
   cd anomaly-detection
   ```

2. **Create a virtual environment & install dependencies**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

---

## Model Training

To train the autoencoder model on your own dataset or MVTec AD dataset:
- Run the core script:
  ```bash
  python code/datascience_project.py
  ```
- Or open `code/datascience_project_collab.ipynb` in Google Colab for GPU-accelerated training.

---

## Running the Gradio Application

To launch the interactive visual inspection UI locally:
```bash
python app.py
```
Open the generated local URL (e.g., `http://127.0.0.1:7860`) in your web browser.

> **Note on Model Checkpoint**: Place your trained weights (`autoencoder_mvtec.pth`) in the root project directory before launching `app.py`.

---

## Deploying to Hugging Face Spaces

1. Create a new Space on [Hugging Face Spaces](https://huggingface.co/new-space) selecting **Gradio** as the SDK.
2. Push `app.py`, `requirements.txt`, and your trained model checkpoint (`autoencoder_mvtec.pth`) to the Space repository.

---

## License

This project is open-source and available under the [MIT License](LICENSE).
