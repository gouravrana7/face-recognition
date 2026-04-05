# Face Recognition

![Python](https://img.shields.io/badge/python-3.12-blue?logo=python)
![OpenCV](https://img.shields.io/badge/opencv--contrib-4.x-green?logo=opencv)
![uv](https://img.shields.io/badge/package_manager-uv-violet)

A real-time face recognition system using OpenCV's **LBPH (Local Binary Pattern Histogram)** algorithm and a Haar cascade face detector. Collect face samples from your webcam, train the model, and run live recognition — all in two commands.

---

## Features

- Webcam-based face data collection (50 samples per person)
- LBPH face recognition with configurable confidence threshold
- Live bounding-box overlay with name and confidence score
- Automatic detection logging to `recognition_log.xlsx`
- Unknown-person detection with snapshot saved to `unKnown.jpg`

---

## Requirements

- Python **3.12+**
- [uv](https://docs.astral.sh/uv/) package manager
- A working **webcam**
- `opencv-contrib-python` (includes the `cv2.face` module — plain `opencv-python` will not work)

---

## Setup

```bash
# 1. Clone the repository
git clone <repo-url>
cd face-recognition

# 2. Install dependencies (creates .venv automatically)
uv sync
```

---

## Usage

### Step 1 — Collect training images

```bash
uv run python main.py
```

- Enter your name when prompted.
- Look at the webcam — it captures **50 face crops** and saves them under `dataset/<your_name>/`.
- Press **ESC** to stop early.

Repeat for every person you want to recognise.

### Step 2 — Train and recognise

```bash
uv run python faceRecognition.py
```

- The model trains on all images in `dataset/` (takes a few seconds).
- A live webcam window opens:
  - **Known face** — green box with name and confidence score.
  - **Unknown face** — yellow box; after 100 consecutive unknown frames the frame is saved as `unKnown.jpg`.
- Every unique detection is appended to `recognition_log.xlsx`.
- Press **q** or **ESC** to exit.

---

## Project Structure

```
face-recognition/
├── main.py                              # Phase 1: data collection
├── faceRecognition.py                   # Phase 2: training + live recognition
├── haarcascade_frontalface_default.xml  # Haar cascade classifier
├── dataset/                             # Training images (one folder per person)
├── recognition_log.xlsx                 # Auto-generated detection log
├── pyproject.toml                       # uv project manifest
├── uv.lock                              # Locked dependency graph
└── docs/                                # Project reports and presentation
```

---

## Configuration

| Constant | File | Default | Description |
|----------|------|---------|-------------|
| `HAAR_FILE` | `faceRecognition.py` | `haarcascade_frontalface_default.xml` | Cascade classifier path |
| `DATASETS` | `faceRecognition.py` | `dataset` | Training images root |
| `WIDTH`, `HEIGHT` | `faceRecognition.py` | `130, 100` | Face crop dimensions (px) |
| `LOG_FILE` | `faceRecognition.py` | `recognition_log.xlsx` | Detection log path |
| Confidence threshold | `faceRecognition.py` | `800` | Below = known, above = unknown |

---

## How It Works

1. **Detection** — Haar cascade scans each frame for faces.
2. **Training** — LBPH model learns texture histograms for each labelled face.
3. **Prediction** — For each detected face the model returns a label and confidence score; lower score means higher similarity.
4. **Logging** — Each new detection (known or unknown) is appended as a timestamped row in the xlsx log.

---

## Notes

- Do **not** commit `dataset/` images or `recognition_log.xlsx` — they may contain personal data.
- The `unKnown.jpg` snapshot is overwritten on every unknown-person trigger.
- Lighting and camera angle significantly affect LBPH accuracy; collect samples under similar conditions to intended use.
