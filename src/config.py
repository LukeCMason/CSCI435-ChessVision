"""
config.py — All settings in one place.
Change values here to adjust detection behaviour.
"""

import os

# ── Model ─────────────────────────────────────────────────────────────────────
# Pre-trained YOLO11n chess model (Apache-2.0).
# Download: https://huggingface.co/yamero999/chess-piece-detection-yolo11n
_here = os.path.dirname(__file__)
EXTERNAL_MODEL  = os.path.join(_here, "..", "models", "best_finetuned.pt")
BASE_MODEL      = os.path.join(_here, "..", "models", "best.pt")

# Inference: fine-tuned model if available, otherwise the pre-trained chess base
MODEL_PATH      = EXTERNAL_MODEL if os.path.exists(EXTERNAL_MODEL) else BASE_MODEL

# Training: always start from the pre-trained base so each run is reproducible
# (avoids compounding errors from fine-tuning an already fine-tuned model)
TRAIN_BASE_PATH = BASE_MODEL

# ── Detection ─────────────────────────────────────────────────────────────────
CONFIDENCE  = 0.25   # ignore predictions below this score (was 0.50 — too strict)
IOU         = 0.45   # non-max suppression overlap threshold
INPUT_SIZE  = 640    # image size passed to YOLO (640 gives finer detail than 416)

# ── Board display ─────────────────────────────────────────────────────────────
BOARD_SIZE  = 640    # pixel size of the perspective-corrected board image

# ── Video display ─────────────────────────────────────────────────────────────
# Refresh the right-panel board state every this many *processed* frames
VIDEO_BOARD_UPDATE_EVERY = 5
