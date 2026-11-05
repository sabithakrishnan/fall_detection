# Real-Time Human Fall Detection System

An advanced computer vision and deep learning framework built to detect human fall events in video sequences (`.avi` / `.mp4`). This repository tracks the evolution of fall detection methodologies, scaling from basic bounding box heuristics using CNNs and Transformers up to robust skeletal posture tracking with state-of-the-art YOLO models.

## 🚀 Architectural Approaches Covered

### 1. Frame-by-Frame Transformer Baseline (DETR)
* **Model:** Facebook Research DETR (`detr_resnet101`) via PyTorch Hub.
* **Core Logic:** Bounding box aspect-ratio heuristic (Width / Height > 1.1).
* **Limitation:** High computational overhead and prone to target loss when a subject lies flat on the floor, as standard object detectors struggle to locate non-upright body postures.

### 2. Pure YOLO Object Detection (With Temporal Smoothing)
* **Model:** Ultralytics YOLO (`yolo11n.pt`).
* **Core Logic:** Uses frame-by-frame bounding box extraction filtered exclusively for the `person` class. It includes a dynamic lower confidence threshold (`0.35`) and an **Alert Hold Buffer** to handle situations where the model briefly loses the target on the floor.
* **Advantage:** Blazing fast execution speeds compared to DETR. By using a hold timer, the "FALL DETECTED!" alert stays locked on the screen long enough for human operators to read it.

### 3. Geometric Feature Extraction (YOLO-Pose Baseline)
* **Model:** Ultralytics YOLO-Pose (`yolo11n-pose.pt`).
* **Core Logic:** Tracks the physical 17-point human skeleton instead of an outer bounding box. It calculates the mathematical **Spine Angle** vector between the midpoint of the shoulders and the midpoint of the hips relative to the floor plane.
* **Advantage:** Zero lag, instant state recovery, and never loses the target when the subject is horizontal on the ground because the skeletal links remain attached.

### 4. Spatio-Temporal Sequence Tracking (YOLO-Pose + LSTM Memory)
* **Model:** YOLO-Pose integrated with a Custom PyTorch Long Short-Term Memory (LSTM) network.
* **Core Logic:** Collects a rolling temporal buffer of 15 frames (~0.5 seconds of history) mapping all 34 coordinate vectors. It tracks downward velocity drops over time.
* **Advantage:** Prevents false alarms caused by rapid everyday movements (e.g., bending down quickly to tie a shoe or sitting on a chair fast).

---

## 🛠️ Installation & Setup

Ensure you have your environment configured (optimized for Google Colab or local Python 3.10+ environments with CUDA acceleration).

```bash
pip install torch torchvision ultralytics opencv-python numpy scikit-learn matplotlib
```

If utilizing video pipelines inside Jupyter/Colab canvases, `ffmpeg` is required for web-optimized stream transcodes:
```bash
sudo apt-get install ffmpeg
```

---


