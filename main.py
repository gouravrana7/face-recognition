"""Face data collection script.

Captures up to MAX_SAMPLES grayscale face crops from the webcam and saves
them to ``dataset/<name>/``.  Run this script once per person before
training with ``faceRecognition.py``.

Usage:
    uv run python main.py
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

import cv2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

CASCADE_PATH: str = str(Path(__file__).parent / "haarcascade_frontalface_default.xml")
DATASET_ROOT: str = "dataset"
CAM_WIDTH: int = 640
CAM_HEIGHT: int = 480
SCALE_FACTOR: float = 1.3
MIN_NEIGHBORS: int = 5
MAX_SAMPLES: int = 50
WAIT_KEY_DELAY_MS: int = 100
COLOR_BLUE: tuple[int, int, int] = (255, 0, 0)


def _get_validated_name() -> str:
    """Prompt the user for a name and validate it is filesystem-safe."""
    name = input("Please enter your name: ").strip()
    if not name or not re.match(r"^[\w\-]+$", name):
        raise SystemExit(
            "Name must be non-empty and contain only letters, digits, hyphens, or underscores."
        )
    return name


def main() -> None:
    """Open the webcam, collect face samples, and save to dataset/."""
    face_id = _get_validated_name()

    face_detector = cv2.CascadeClassifier(CASCADE_PATH)
    if face_detector.empty():
        raise SystemExit(f"Failed to load cascade classifier from '{CASCADE_PATH}'.")

    person_dir = os.path.join(DATASET_ROOT, face_id)
    os.makedirs(person_dir, exist_ok=True)

    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

    count = 0
    try:
        while True:
            ret, img = cam.read()
            if not ret or img is None:
                logger.error("Failed to capture frame; exiting.")
                break

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, SCALE_FACTOR, MIN_NEIGHBORS)

            for x, y, w, h in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), COLOR_BLUE, 2)
                count += 1
                filename = f"{face_id}_{count}.jpg"
                cv2.imwrite(
                    os.path.join(person_dir, filename),
                    gray[y : y + h, x : x + w],
                )

            cv2.imshow("image", img)
            k = cv2.waitKey(WAIT_KEY_DELAY_MS) & 0xFF
            if k == 27 or count >= MAX_SAMPLES:
                break
    finally:
        cam.release()
        cv2.destroyAllWindows()

    logger.info("Collected %d samples for '%s'.", count, face_id)


if __name__ == "__main__":
    main()
