"""
utils.py — Drawing helpers and board display utilities.
"""

import cv2
import numpy as np


# ── Piece label → FEN character ───────────────────────────────────────────────
# The model uses "white_pawn", "black_queen" etc.
_PIECE_FEN = {
    "pawn": "P", "knight": "N", "bishop": "B",
    "rook": "R", "queen":  "Q", "king":   "K",
}

def label_to_fen(label: str) -> str:
    """Convert a model class name to a FEN character.

    Handles both separator styles used by different model weights:
      underscore: "white_pawn" → "P",  "black_queen" → "q"
      hyphen:     "white-pawn" → "P",  "black-queen" → "q"
    """
    norm     = label.lower().replace("-", "_")   # normalise hyphens → underscores
    is_white = norm.startswith("white_")
    piece    = norm.split("_")[-1]               # "pawn", "rook", etc.
    fen      = _PIECE_FEN.get(piece, "?")
    return fen if is_white else fen.lower()


# ── Drawing ───────────────────────────────────────────────────────────────────

def draw_board_outline(frame: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Draw a green quadrilateral around the detected chessboard."""
    out = frame.copy()
    pts = quad.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(out, [pts], isClosed=True, color=(0, 220, 0), thickness=3)
    return out


def draw_grid(image: np.ndarray) -> np.ndarray:
    """Draw an 8×8 grid over a square board image."""
    out  = image.copy()
    h, w = out.shape[:2]
    sq_w = w // 8
    sq_h = h // 8
    for i in range(1, 8):
        cv2.line(out, (i * sq_w, 0), (i * sq_w, h), (0, 220, 220), 1)
        cv2.line(out, (0, i * sq_h), (w, i * sq_h), (0, 220, 220), 1)
    cv2.rectangle(out, (0, 0), (w - 1, h - 1), (0, 200, 0), 2)
    return out


def draw_detections(image: np.ndarray, detections: list) -> np.ndarray:
    """Draw bounding boxes and labels for each detected piece.

    Black pieces use dark green overlays; white pieces use bright green.
    This makes it visually easy to distinguish the two sets at a glance.
    """
    out = image.copy()
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        label = det["label"]
        conf  = det["conf"]
        text  = f"{label} {conf:.2f}"

        # Dark green for black pieces, bright green for white pieces
        tid      = det.get("track_id")
        is_black = label.lower().startswith("black")
        color    = (0, 120, 0) if is_black else (0, 255, 0)
        txt_col  = (210, 210, 210) if is_black else (0, 0, 0)
        if tid is not None:
            text = f"#{tid} {text}"

        # Box
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        # Label background + text
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        ty = max(y1 - 4, th + 4)
        cv2.rectangle(out, (x1, ty - th - 4), (x1 + tw + 2, ty + 2), color, -1)
        cv2.putText(out, text, (x1 + 1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, txt_col, 1)
    return out


# ── Board state display ───────────────────────────────────────────────────────

def board_to_fen(board: list) -> str:
    """Convert an 8×8 board matrix to a FEN string."""
    rows = []
    for row in board:
        empty = 0
        row_str = ""
        for cell in row:
            if cell == "":
                empty += 1
            else:
                if empty:
                    row_str += str(empty)
                    empty = 0
                row_str += cell
        if empty:
            row_str += str(empty)
        rows.append(row_str)
    return "/".join(rows) + " w - - 0 1"


def board_to_text(board: list) -> str:
    """Return the board as a plain-text table for display."""
    lines = ["   a  b  c  d  e  f  g  h"]
    for i, row in enumerate(board):
        rank = 8 - i
        cells = [f" {c if c else '.'} " for c in row]
        lines.append(f"{rank} {''.join(cells)}")
    return "\n".join(lines)
