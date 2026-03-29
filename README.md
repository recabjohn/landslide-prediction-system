# 🛰️ AI-Based Landslide Prediction and Detection System
### Using YOLOv11 and Satellite Imagery

<p align="center">
  <strong>An end-to-end deep learning system for automated landslide detection and risk assessment from satellite imagery</strong>
</p>

---

## 📋 Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Docker Hosting](#docker-hosting)
- [Azure Deployment (Docker)](#azure-deployment-docker)
- [Dataset Preparation](#dataset-preparation)
- [Training Your Own Model](#training-your-own-model)
- [Project Structure](#project-structure)
- [Modules](#modules)
- [References](#references)

---

## 🔍 Overview

This project implements an **AI-based landslide prediction and detection system** that uses **YOLOv11 object detection** on **satellite imagery** to:

1. **Detect** landslides, debris flows, and terrain anomalies in satellite images
2. **Predict** landslide risk levels using detection analysis and terrain feature extraction
3. **Visualize** results with annotated bounding boxes and risk overlays
4. **Display** everything through an interactive **Streamlit web dashboard**

> Designed as a final-year engineering project prototype.

---

## 🏗️ System Architecture

```
Satellite Image (Upload / API / Demo)
          │
          ▼
  ┌───────────────────┐
  │ Image Preprocessing│ ← Resize, normalize, color convert
  └────────┬──────────┘
           │
           ▼
  ┌───────────────────┐
  │ YOLOv11 Inference  │ ← Object detection model
  └────────┬──────────┘
           │
           ▼
  ┌───────────────────┐
  │ Landslide Detection│ ← Bounding boxes + confidence scores
  └────────┬──────────┘
           │
           ▼
  ┌───────────────────┐
  │ Risk Prediction   │ ← Composite risk score (0-100%)
  └────────┬──────────┘
           │
           ▼
  ┌───────────────────┐
  │ Visualization + UI│ ← Streamlit dashboard
  └───────────────────┘
```

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔍 **YOLO Detection** | YOLOv11-based landslide and debris flow detection |
| 📊 **Risk Scoring** | 0-100% risk score with Low/Medium/High/Critical levels |
| 🛰️ **Satellite Integration** | Sentinel-2 API integration with local fallback |
| 🖼️ **Visual Overlays** | Bounding boxes, confidence labels, risk badges |
| 🌐 **Web Dashboard** | Interactive Streamlit UI with dark theme |
| 🎮 **Demo Mode** | Works out-of-the-box with generated sample images |
| 🏋️ **Training Pipeline** | Complete YOLO training script for custom datasets |

---

## ⚙️ Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager
- (Optional) NVIDIA GPU with CUDA for training

### Setup

```bash
# Clone or navigate to the project
cd /path/to/project

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# venv\Scripts\activate    # On Windows

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Quick Start

### Run the Dashboard

```bash
# Launch the Streamlit app
streamlit run ui/app.py
```

The dashboard opens at `http://localhost:8501`.

---

## 🐳 Docker Hosting

You can run the full Streamlit dashboard in Docker locally.

### Option A: Docker CLI

```bash
docker build -t landslide-dashboard .
docker run --rm -p 8501:8501 landslide-dashboard
```

Open: `http://localhost:8501`

### Option B: Docker Compose

```bash
docker compose up --build
```

Open: `http://localhost:8501`

---

## ☁️ Azure Deployment (Docker)

This project includes an Azure App Service deployment script for the Docker image.

### Prerequisites

- Azure account and active subscription
- [Azure CLI](https://learn.microsoft.com/cli/azure/install-azure-cli)
- Docker Desktop running

### Deploy with one script

```bash
chmod +x scripts/deploy_azure_appservice.sh

./scripts/deploy_azure_appservice.sh \
  <resource-group> \
  <location> \
  <acr-name> \
  <app-service-plan> \
  <webapp-name>
```

Example:

```bash
./scripts/deploy_azure_appservice.sh \
  landslide-rg \
  centralindia \
  landslideacr123 \
  landslide-plan \
  landslide-dashboard-2026
```

Notes:
- `acr-name` must be globally unique and use lowercase letters/numbers.
- `webapp-name` must be globally unique.
- The app is exposed on port `8501` via `WEBSITES_PORT=8501`.

### Demo Mode (No setup required)

1. Launch the dashboard
2. Toggle **"Demo Mode"** in the sidebar
3. Click **"🚀 Run Detection"**
4. View detection results with bounding boxes and risk analysis

### Upload Your Own Image

1. Launch the dashboard
2. Upload a satellite image via the sidebar
3. Adjust the **confidence threshold** slider
4. Click **"🚀 Run Detection"**

---

## ☁️ Cloud Deployment (Free Hosting)

You can easily host this dashboard online for free using **Streamlit Community Cloud**, making it perfect for sharing your final year project presentation.

1. **Push to GitHub**:
   - Create a free GitHub account.
   - Initialize Git in this folder and push all files to a new public or private repository.
   ```bash
   git init
   git add .
   git commit -m "Initial commit for landslide project"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   git push -u origin main
   ```

2. **Deploy on Streamlit**:
   - Go to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
   - Click **"New app"**.
   - Select your new repository and branch (`main`).
   - For **Main file path**, enter: `ui/app.py`
   - Click **Deploy!**

*Note: The `requirements.txt` has already been optimized for Linux cloud deployment (using `opencv-python-headless`). Streamlit will automatically install PyTorch, YOLO, and all dependencies.*

---

## 📦 Dataset Preparation

### Recommended Datasets

| Dataset | Source | Description |
|---------|--------|-------------|
| **Landslide4Sense** | [IARAI](https://www.iarai.ac.at/landslide4sense/) | Benchmark satellite dataset |
| **NASA Landslide Catalog** | [NASA](https://data.nasa.gov/) | Global landslide inventory |
| **Kaggle Landslide** | [Kaggle](https://www.kaggle.com/) | Various landslide image datasets |

### Dataset Structure

```
dataset/
├── train/
│   ├── images/    # Training satellite images
│   └── labels/    # YOLO format annotations
├── val/
│   ├── images/    # Validation images
│   └── labels/    # Validation annotations
└── test/
    ├── images/    # Test images
    └── labels/    # Test annotations
```

### Annotation Format (YOLO)

Each `.txt` label file contains one line per object:
```
<class_id> <x_center> <y_center> <width> <height>
```
All values normalized to [0, 1].

**Classes:**
- `0` — landslide
- `1` — debris_flow
- `2` — normal_terrain

See [ANNOTATION_GUIDE.md](ANNOTATION_GUIDE.md) for detailed annotation instructions.

---

## 🏋️ Training Your Own Model

```bash
# Using the training script
python train.py --data dataset.yaml --model yolo11n.pt --epochs 50 --imgsz 640

# Or using YOLO CLI directly
yolo detect train data=dataset.yaml model=yolo11n.pt epochs=50 imgsz=640
```

Place your trained weights in the `models/` directory.

---

## 📁 Project Structure

```
project/
│
├── dataset/                    # Dataset directory
│   ├── train/images/
│   ├── train/labels/
│   ├── val/images/
│   ├── val/labels/
│   ├── test/images/
│   └── test/labels/
│
├── models/                     # Trained model weights
│
├── satellite_fetcher/          # Module 1: Satellite image fetching
│   ├── __init__.py
│   └── fetch_images.py
│
├── preprocessing/              # Module 2: Image preprocessing
│   ├── __init__.py
│   └── preprocess.py
│
├── detection/                  # Module 5: YOLO inference
│   ├── __init__.py
│   └── detect_landslides.py
│
├── prediction/                 # Module 6: Risk estimation
│   ├── __init__.py
│   └── risk_estimator.py
│
├── visualization/              # Module 7: Result visualization
│   ├── __init__.py
│   └── draw_boxes.py
│
├── ui/                         # Module 8: Streamlit dashboard
│   ├── __init__.py
│   └── app.py
│
├── demo/                       # Module 9: Demo mode
│   ├── __init__.py
│   ├── generate_samples.py
│   └── sample_results.json
│
├── train.py                    # Module 4: YOLO training script
├── dataset.yaml                # YOLO dataset config
├── ANNOTATION_GUIDE.md         # Annotation instructions
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## 🧩 Modules

### 1. Satellite Image Fetcher (`satellite_fetcher/`)
Fetches satellite imagery via Sentinel-2 API or loads local images. Includes demo image fallback.

### 2. Image Preprocessing (`preprocessing/`)
Resizes, normalizes, and converts images for YOLO inference (640×640 input).

### 3. YOLO Training (`train.py`)
Complete YOLOv11 training pipeline with configurable hyperparameters.

### 4. Detection (`detection/`)
Runs YOLO inference on satellite images and extracts bounding boxes with confidence scores.

### 5. Risk Prediction (`prediction/`)
Analyzes detections and terrain characteristics to compute a risk score (0-100%).

### 6. Visualization (`visualization/`)
Draws annotated bounding boxes, confidence labels, and risk overlay on images.

### 7. Streamlit Dashboard (`ui/`)
Interactive web UI for image upload, detection, and risk visualization.

---

## 📚 References

1. Jocher, G., et al. (2023). *Ultralytics YOLOv11*. https://github.com/ultralytics/ultralytics
2. Ghorbanzadeh, O., et al. (2022). *Landslide4Sense: Reference Benchmark Data and Deep Learning Models for Landslide Detection*. IEEE TGRS.
3. Kirschbaum, D., et al. (2015). *A global landslide catalog for hazard applications*. Natural Hazards.
4. European Space Agency. *Sentinel-2 Mission*. https://sentinel.esa.int/web/sentinel/missions/sentinel-2

---

## 📄 License

This project is developed for academic and educational purposes.

---

<p align="center">
  <em>Built with ❤️ for Final Year Engineering Project</em>
</p>
