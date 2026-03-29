# 📝 Landslide Image Annotation Guide

This guide explains how to annotate satellite images for training the YOLOv11 landslide detection model.

---

## 📋 Classes to Annotate

| Class ID | Class Name | Description | Visual Cues |
|----------|------------|-------------|-------------|
| 0 | `landslide` | Active or recent landslide scar | Bare earth, exposed soil, disrupted terrain, fan-shaped debris |
| 1 | `debris_flow` | Debris/mud flow channel | Narrow channel of displaced material, often following a watercourse |
| 2 | `normal_terrain` | Stable terrain feature | Undisturbed vegetation, stable slopes, roads, structures |

---

## 🔧 Annotation Tools

### Option 1: LabelImg (Desktop — Recommended for Beginners)

**Install:**
```bash
pip install labelImg
```

**Steps:**
1. Launch: `labelImg`
2. Click **"Open Dir"** → select `dataset/train/images/`
3. Set save format to **YOLO** (click the format button until it says "YOLO")
4. Set **"Save Dir"** → `dataset/train/labels/`
5. Press **W** to draw a bounding box
6. Enter the class name and click OK
7. Press **D** to move to the next image
8. Press **Ctrl+S** to save

**Keyboard shortcuts:**
- `W` — Create a bounding box
- `D` — Next image
- `A` — Previous image
- `Del` — Delete selected box
- `Ctrl+S` — Save

---

### Option 2: Roboflow (Cloud-Based — Best for Teams)

1. Go to [https://roboflow.com](https://roboflow.com) and create a free account
2. Create a new project → Select **"Object Detection"**
3. Upload your satellite images
4. Define classes: `landslide`, `debris_flow`, `normal_terrain`
5. Draw bounding boxes around each feature
6. When done, click **"Generate"** → **"Export"** → Choose **YOLOv11** format
7. Download and extract into `dataset/` directory

**Advantages:** Auto-augmentation, dataset versioning, team collaboration

---

### Option 3: CVAT (Self-Hosted — Best for Large Datasets)

1. Install CVAT via Docker:
   ```bash
   git clone https://github.com/opencv/cvat.git
   cd cvat
   docker compose up -d
   ```
2. Open `http://localhost:8080` in your browser
3. Create a new task → Upload satellite images
4. Add labels: `landslide`, `debris_flow`, `normal_terrain`
5. Annotate using the rectangle tool
6. Export annotations → Select **"YOLO 1.1"** format
7. Place the files in `dataset/train/labels/` or `dataset/val/labels/`

---

## 📐 YOLO Annotation Format

Each image needs a corresponding `.txt` label file with the **same filename**.

**Example:**
```
Image: dataset/train/images/satellite_001.jpg
Label: dataset/train/labels/satellite_001.txt
```

**Label file format** (one line per object):
```
<class_id> <x_center> <y_center> <width> <height>
```

All values are **normalized** (0.0 to 1.0) relative to image dimensions.

**Example label file content:**
```
0 0.4500 0.5200 0.3000 0.2500
1 0.7000 0.8000 0.1500 0.1000
2 0.2000 0.3000 0.1000 0.0800
```

**Conversion formula:**
```
x_center = (x_left + x_right) / 2 / image_width
y_center = (y_top + y_bottom) / 2 / image_height
width    = (x_right - x_left) / image_width
height   = (y_bottom - y_top) / image_height
```

---

## 🎯 Annotation Best Practices

1. **Tight bounding boxes** — Draw boxes that closely fit the landslide boundary
2. **Include the entire feature** — Don't crop out parts of a landslide
3. **Multiple scales** — Annotate both large and small features
4. **Consistency** — Apply the same criteria across all images
5. **Minimum 200 images** — Aim for at least 200 annotated images for decent results
6. **Balanced classes** — Try to have approximately equal representation per class
7. **80/20 split** — Put 80% in `train/` and 20% in `val/`

---

## 📁 Expected Directory Structure After Annotation

```
dataset/
├── train/
│   ├── images/
│   │   ├── satellite_001.jpg
│   │   ├── satellite_002.png
│   │   └── ...
│   └── labels/
│       ├── satellite_001.txt
│       ├── satellite_002.txt
│       └── ...
├── val/
│   ├── images/
│   │   ├── satellite_100.jpg
│   │   └── ...
│   └── labels/
│       ├── satellite_100.txt
│       └── ...
└── test/
    ├── images/
    └── labels/
```

---

## ✅ Validation Checklist

- [ ] Each image has a matching `.txt` label file
- [ ] Label files use correct class IDs (0, 1, or 2)
- [ ] All bounding box values are between 0.0 and 1.0
- [ ] No empty label files for images with objects
- [ ] Images are in `.jpg`, `.jpeg`, or `.png` format
- [ ] `dataset.yaml` paths are correct
