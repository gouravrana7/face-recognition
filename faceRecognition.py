"""Real-time face recognition using OpenCV's LBPH algorithm.

Loads grayscale face images from ``dataset/`` (populated by ``main.py``),
trains an LBPH model, and runs a live webcam recognition loop.
Detected persons are logged to ``recognition_log.xlsx``.

Usage:
    uv run python faceRecognition.py

Press **q** or **ESC** to exit.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from openpyxl import Workbook, load_workbook

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# --- Paths ---
HAAR_FILE: str = str(Path(__file__).parent / "haarcascade_frontalface_default.xml")
DATASETS_DIR: str = "dataset"
LOG_FILE: str = "recognition_log.xlsx"

# --- Face crop dimensions (px) ---
FACE_WIDTH: int = 130
FACE_HEIGHT: int = 100

# --- Detection / recognition parameters ---
SCALE_FACTOR: float = 1.3
MIN_NEIGHBORS: int = 5
CONFIDENCE_THRESHOLD: float = 800.0
UNKNOWN_FRAME_THRESHOLD: int = 100
WAIT_KEY_DELAY_MS: int = 10
TEXT_OFFSET: int = 10

# --- Display colours (BGR) ---
COLOR_GREEN: tuple[int, int, int] = (0, 255, 0)
COLOR_RED: tuple[int, int, int] = (0, 0, 255)
COLOR_CYAN: tuple[int, int, int] = (0, 255, 255)
RECT_THICKNESS: int = 3


def append_log(name: str, confidence: float | None, log_path: str = LOG_FILE) -> None:
    """Append a detection event to the xlsx log, creating it if necessary."""
    try:
        if not os.path.exists(log_path):
            wb = Workbook()
            ws = wb.active
            ws.title = "log"
            ws.append(["timestamp", "name", "confidence"])
        else:
            wb = load_workbook(log_path)
            ws = wb.active
        ws.append([
            datetime.now().isoformat(timespec="seconds"),
            name,
            confidence if confidence is not None else "",
        ])
        wb.save(log_path)
    except OSError as exc:
        logger.error("Failed to write recognition log: %s", exc)


def _load_training_data(
    datasets_dir: str,
) -> tuple[np.ndarray, np.ndarray, dict[int, str]]:
    """Load all grayscale face images and labels from *datasets_dir*."""
    images: list[np.ndarray] = []
    labels: list[int] = []
    names: dict[int, str] = {}

    for person_id, subdir in enumerate(sorted(os.listdir(datasets_dir))):
        subject_path = os.path.join(datasets_dir, subdir)
        if not os.path.isdir(subject_path):
            continue
        names[person_id] = subdir
        for filename in os.listdir(subject_path):
            path = os.path.join(subject_path, filename)
            raw_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if raw_img is None:
                continue
            resized_img = cv2.resize(raw_img, (FACE_WIDTH, FACE_HEIGHT))
            images.append(resized_img)
            labels.append(person_id)

    return np.array(images), np.array(labels), names


def _run_recognition_loop(
    webcam: cv2.VideoCapture,
    model: cv2.face.LBPHFaceRecognizer,  # type: ignore[name-defined]
    face_cascade: cv2.CascadeClassifier,
    names: dict[int, str],
) -> None:
    """Run the live recognition loop until the user presses q or ESC."""
    logged_once = False
    unknown_frame_count = 0

    while True:
        ret, frame = webcam.read()
        if not ret or frame is None:
            logger.error("Failed to capture frame; exiting.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, SCALE_FACTOR, MIN_NEIGHBORS)

        for x, y, w, h in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), COLOR_GREEN, RECT_THICKNESS)
            resized_face = cv2.resize(
                gray[y : y + h, x : x + w], (FACE_WIDTH, FACE_HEIGHT)
            )
            label, raw_conf = model.predict(resized_face)
            conf = float(raw_conf)

            if conf < CONFIDENCE_THRESHOLD:
                name = names[label]
                cv2.putText(
                    frame,
                    f"{name} - {conf:.0f}",
                    (x - TEXT_OFFSET, y - TEXT_OFFSET),
                    cv2.FONT_HERSHEY_PLAIN,
                    2,
                    COLOR_RED,
                )
                logger.info("Recognised: %s (conf=%.0f)", name, conf)
                if not logged_once:
                    append_log(name, conf)
                    logged_once = True
                unknown_frame_count = 0
            else:
                unknown_frame_count += 1
                cv2.putText(
                    frame,
                    "Unknown",
                    (x - TEXT_OFFSET, y - TEXT_OFFSET),
                    cv2.FONT_HERSHEY_PLAIN,
                    1,
                    COLOR_GREEN,
                )
                if unknown_frame_count > UNKNOWN_FRAME_THRESHOLD:
                    logger.warning("Unknown person detected.")
                    if not logged_once:
                        append_log("Unknown", None)
                        logged_once = True
                    cv2.imwrite("unKnown.jpg", frame)
                    unknown_frame_count = 0

        cv2.putText(
            frame,
            "Press q or ESC to exit",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            COLOR_CYAN,
            2,
        )
        cv2.imshow("FaceRecognition", frame)
        key = cv2.waitKey(WAIT_KEY_DELAY_MS) & 0xFF
        if key in (27, ord("q")):
            break


def main() -> None:
    """Validate environment, train LBPH model, run recognition loop."""
    if not hasattr(cv2, "face"):
        raise SystemExit(
            "OpenCV was built without the face module. "
            "Install opencv-contrib-python and try again."
        )

    face_cascade = cv2.CascadeClassifier(HAAR_FILE)
    if face_cascade.empty():
        raise SystemExit(
            f"Failed to load Haar cascade from '{HAAR_FILE}'. Check the file path."
        )

    logger.info("Training model...")
    images, labels, names = _load_training_data(DATASETS_DIR)

    if len(images) == 0:
        raise SystemExit(
            "No training images found in dataset. Run main.py to create some first."
        )

    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(images, labels)
    logger.info("Training complete (%d images, %d persons).", len(images), len(names))

    webcam = cv2.VideoCapture(0)
    try:
        _run_recognition_loop(webcam, model, face_cascade, names)
    finally:
        webcam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
