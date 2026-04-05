# Face Recognition — Claude Code Context

## Project Overview

A two-phase real-time face recognition system built with OpenCV's LBPH (Local Binary Pattern Histogram) algorithm.

- **Phase 1 — Data Collection** (`main.py`): captures 50 grayscale face images per person from webcam, saves to `dataset/<name>/`.
- **Phase 2 — Recognition** (`faceRecognition.py`): loads the dataset, trains the LBPH model, runs live webcam recognition, and appends every detection event to `recognition_log.xlsx`.

## Architecture

```
face-recognition/
├── main.py                          # Data-collection script (run first)
├── faceRecognition.py               # Training + live-recognition script
├── haarcascade_frontalface_default.xml  # Haar cascade for face detection
├── dataset/                         # Training images, one sub-dir per person
│   └── <person_name>/               # e.g. dataset/abhishek/
│       └── <name>_<n>.jpg           # grayscale face crops (50 per person)
├── recognition_log.xlsx             # Append-only detection log
├── docs/                            # Project reports and slides (binary)
├── pyproject.toml                   # uv project manifest
├── uv.lock                          # Locked dependency graph
├── .python-version                  # Pins Python 3.12
└── CLAUDE.md                        # This file
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.12 |
| Package manager | **uv** (pyproject.toml + uv.lock) |
| Computer vision | `opencv-contrib-python` (LBPH, Haar cascade) |
| Numerical | `numpy` |
| Logging | `openpyxl` (xlsx append-log) |

## Key Design Decisions

- **LBPH recognizer** — chosen for robustness to lighting variation; threshold `< 800` means "known face".
- **Per-session `logged_once` flag** — prevents duplicate log rows within a single run.
- **Unknown person threshold** — after 100 consecutive unknown frames the frame is saved as `unKnown.jpg` and logged.
- **Image dimensions** — all face crops are resized to 130 × 100 px before training and prediction.

## Coding Conventions

- **PEP 8** style; formatted with **black**, linted with **ruff**.
- **Type annotations** on every function signature (`from __future__ import annotations` for 3.12 compat).
- Immutable data patterns — prefer returning new objects over mutation.
- Functions < 50 lines; files < 800 lines.
- No `print()` in library code — use the `logging` module.
- Constants at module level in `UPPER_SNAKE_CASE`.
- Validate inputs at system boundaries (webcam, file I/O).

## Setup & Run

```bash
# Install dependencies with uv
uv sync

# Step 1 — collect training images (runs webcam, captures 50 frames, press ESC to stop early)
uv run python main.py

# Step 2 — train model and start live recognition (press q or ESC to exit)
uv run python faceRecognition.py
```

**Prerequisites:** a working webcam and `opencv-contrib-python` (not the plain `opencv-python`).

## Development Notes

- `haarcascade_frontalface_default.xml` must exist in the working directory (or update the path constant).
- `dataset/` must contain at least one person sub-directory with images before running `faceRecognition.py`.
- `recognition_log.xlsx` is created automatically on first detection.
- `unKnown.jpg` is overwritten each time an unknown person triggers the threshold.
- Do **not** commit `dataset/` images or `recognition_log.xlsx` — they may contain PII.

## Testing

```bash
uv run pytest --cov=. --cov-report=term-missing
```

Target: **80 % coverage** minimum. Use `pytest.mark.unit` / `pytest.mark.integration`.
