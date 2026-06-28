"""
chess_vision.py — The six core pipeline functions.

Pipeline order:
    frame       = load_input(source)
    quad        = detect_board(frame)
    board_img   = rectify_board(frame, quad)
    detections  = detect_pieces(frame)           <- always run on original frame
    board_state = create_board_state(detections, frame, quad)
    ann_frame, ann_board = draw_results(frame, quad, board_img, detections)
"""

import cv2
import numpy as np
from ultralytics import YOLO

import config
from utils import label_to_fen, draw_board_outline, draw_grid, draw_detections

# ── Load the YOLO model once at import time ───────────────────────────────────
model = YOLO(config.MODEL_PATH)


# ── Preprocessing helper ──────────────────────────────────────────────────────

def _clahe_color(img: np.ndarray) -> np.ndarray:
    """Apply CLAHE to the L channel (LAB space) to normalise contrast.

    Using LAB ensures hue and saturation are untouched — critical because
    black vs white piece discrimination depends entirely on brightness/colour.
    Grayscale conversion must NOT be used before YOLO inference for this reason.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def _verify_piece_color(enhanced: np.ndarray, box: list, label: str,
                        board_median: float) -> str:
    """Cross-check the model's colour prediction against actual pixel brightness.

    Uses two complementary rules so the check works in all four combinations
    of piece-colour × square-colour:

    Rule A — relative (piece vs its own square background):
      Sample the four corner patches of the bounding box as an estimate of
      the square colour behind the piece.  If the piece centre is much
      brighter than its square it is white; much darker it is black.
      This handles the main failure mode: white piece on dark square that
      the model called 'black' because low overall brightness.

    Rule B — absolute extremes (clearly wrong predictions):
      p75 < 90  → even the top-quarter of pixels is very dark → black piece.
      p25 > 175 → even the bottom-quarter is very bright    → white piece.

    Only one rule needs to fire for a correction; the model's label is
    trusted in ambiguous (mid-range) cases.
    """
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    if bw < 6 or bh < 6:
        return label

    gray_img = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)

    norm          = label.lower().replace("-", "_")
    is_white_pred = norm.startswith("white_")
    piece_type    = norm.split("_", 1)[1]   # e.g. 'bishop', 'king', 'pawn'

    # ── Rule A: local square colour from corner patches ──────────────────────
    cs = max(3, min(bw, bh) // 7)   # corner-sample size
    corner_patches = [
        gray_img[y1        : y1 + cs, x1        : x1 + cs],
        gray_img[y1        : y1 + cs, x2 - cs   : x2     ],
        gray_img[y2 - cs   : y2,      x1        : x1 + cs],
        gray_img[y2 - cs   : y2,      x2 - cs   : x2     ],
    ]
    corner_px = np.concatenate([c.ravel() for c in corner_patches if c.size > 0])
    sq_color  = float(np.median(corner_px)) if corner_px.size > 0 else board_median

    # Inner 50 % crop — piece body with less background contamination
    px, py    = bw // 4, bh // 4
    inner     = gray_img[y1 + py : y2 - py, x1 + px : x2 - px]
    if inner.size == 0:
        return label
    gray_flat    = inner.ravel().astype(np.float32)
    piece_center = float(np.median(gray_flat))

    _REL = 25   # brightness gap (piece vs square) needed to trigger a correction

    # ── Rule B: absolute extremes ─────────────────────────────────────────────
    p25 = float(np.percentile(gray_flat, 25))
    p75 = float(np.percentile(gray_flat, 75))
    _ABS_DARK   = 90    # p75 below this → definitely a black piece
    _ABS_BRIGHT = 150   # p25 above this → definitely a white piece

    # ── Apply ─────────────────────────────────────────────────────────────────
    if is_white_pred:
        if (piece_center < sq_color - _REL) or (p75 < _ABS_DARK):
            return f"black-{piece_type}"
    else:
        if (piece_center > sq_color + _REL) or (p25 > _ABS_BRIGHT):
            return f"white-{piece_type}"

    return label


# ── 1. Load input ─────────────────────────────────────────────────────────────

def load_input(source) -> np.ndarray:
    """Load an image from a file path, numpy array, or bytes object.

    Returns a BGR numpy array ready for OpenCV.
    """
    if isinstance(source, np.ndarray):
        return source

    if isinstance(source, (bytes, bytearray)):
        arr = np.frombuffer(source, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    # Assume file path string
    return cv2.imread(str(source))


# ── 2. Detect chessboard ──────────────────────────────────────────────────────

def detect_board(frame: np.ndarray) -> np.ndarray | None:
    """Find the chessboard using Canny edges and contour detection.

    Returns a (4,2) float32 array [TL, TR, BR, BL], or None if not found.
    Accepts only quads that are:
      - 5%–95% of image area
      - convex
      - aspect ratio 0.65–1.50 (roughly square)
      - all 4 interior angles between 50° and 130°
      - not the exact image border (corners not all within 5% of an edge)
    Two edge-detection passes are tried: low-sensitivity first (misses noise),
    then high-sensitivity (catches faint board outlines in real photos).
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # CLAHE improves contrast in real-photo conditions before Canny
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray  = clahe.apply(gray)
    # Bilateral preserves sharp board/piece edges while removing noise (Lab 5 analysis)
    blur  = cv2.bilateralFilter(gray, 9, 75, 75)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    img_h, img_w = frame.shape[:2]
    image_area   = img_h * img_w

    # Only reject a quad whose corners are ALL within 5% of any image edge
    # (catches only the literal image-border rectangle; allows large boards).
    edge_thresh_x = img_w * 0.05
    edge_thresh_y = img_h * 0.05

    def _near_edge(x, y):
        return (x < edge_thresh_x or x > img_w - edge_thresh_x or
                y < edge_thresh_y or y > img_h - edge_thresh_y)

    def _try_find(edges):
        edges = cv2.dilate(edges, kernel, iterations=1)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:60]:
            area = cv2.contourArea(contour)
            if not (image_area * 0.03 < area < image_area * 0.97):
                continue

            # --- quad approximation ------------------------------------------
            # First try the convex hull: absorbs UI overlays / chat bubbles /
            # partial occlusions that add a protrusion to one edge and prevent
            # approxPolyDP from returning exactly 4 vertices on the raw contour.
            hull      = cv2.convexHull(contour)
            perimeter = cv2.arcLength(hull, True)
            approx    = cv2.approxPolyDP(hull, 0.02 * perimeter, True)
            if len(approx) != 4:
                # Fall back to raw contour with a more tolerant epsilon
                perimeter = cv2.arcLength(contour, True)
                approx    = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
                if len(approx) != 4:
                    continue
            # -----------------------------------------------------------------

            if not cv2.isContourConvex(approx):
                continue
            pts = approx.reshape(4, 2).astype("float32")
            if all(_near_edge(x, y) for x, y in pts):
                continue
            x, y, w, h = cv2.boundingRect(approx)
            aspect = w / h if h > 0 else 0
            if not (0.65 < aspect < 1.50):
                continue
            if not _is_rectangular(pts):
                continue
            ordered = _order_corners(pts)
            if not _is_checkerboard(frame, ordered):
                continue
            return ordered
        return None

    # Pass 1: conservative Canny (less noise, cleaner boards)
    result = _try_find(cv2.Canny(blur, 30, 120))
    if result is not None:
        return result

    # Pass 2: sensitive Canny (picks up faint edges in real photos)
    result = _try_find(cv2.Canny(blur, 10, 60))
    return result


def _is_rectangular(pts: np.ndarray) -> bool:
    """Return True if all 4 interior angles of the quad are 50°–130°."""
    for i in range(4):
        p1 = pts[i]
        p2 = pts[(i + 1) % 4]
        p3 = pts[(i + 2) % 4]
        v1 = p1 - p2
        v2 = p3 - p2
        norm = np.linalg.norm(v1) * np.linalg.norm(v2)
        if norm < 1e-8:
            return False
        cos_a = np.dot(v1, v2) / norm
        angle = np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0)))
        if not (50 < angle < 130):
            return False
    return True


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Sort 4 points into [top-left, top-right, bottom-right, bottom-left]."""
    ordered = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    ordered[1] = pts[np.argmin(d)]
    ordered[3] = pts[np.argmax(d)]
    return ordered

def _is_checkerboard(frame: np.ndarray, pts: np.ndarray) -> bool:
    """Return True if the quad region has a chess-board alternating pattern.

    Warps the candidate region to 640×640, divides into 8×8 cells, and checks
    that even-indexed cells (r+c even) vs odd-indexed cells differ in mean
    brightness by at least 12 units.  This rejects false-positive rectangles
    such as phone frames, picture frames, and table edges that pass the
    geometric checks but lack the alternating light/dark square pattern.
    """
    sq   = 80          # 640 // 8
    size = sq * 8      # 640
    dst  = np.array(
        [[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]],
        dtype="float32",
    )
    M    = cv2.getPerspectiveTransform(pts, dst)
    gray = cv2.cvtColor(cv2.warpPerspective(frame, M, (size, size)), cv2.COLOR_BGR2GRAY)

    even_means, odd_means = [], []
    for r in range(8):
        for c in range(8):
            cell = float(gray[r * sq : (r + 1) * sq, c * sq : (c + 1) * sq].mean())
            (even_means if (r + c) % 2 == 0 else odd_means).append(cell)

    return abs(np.mean(even_means) - np.mean(odd_means)) > 12

# ── 3. Perspective correction ─────────────────────────────────────────────────

def rectify_board(frame: np.ndarray, quad: np.ndarray | None) -> np.ndarray:
    """Warp the board region into a square top-down view.

    If quad is None, returns the original frame resized to BOARD_SIZE.
    """
    size = config.BOARD_SIZE

    if quad is None:
        return cv2.resize(frame, (size, size))

    dst = np.array([[0, 0], [size - 1, 0],
                    [size - 1, size - 1], [0, size - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(frame, M, (size, size))


# ── 4. Detect pieces ──────────────────────────────────────────────────────────

def detect_pieces(frame: np.ndarray, board_img: np.ndarray | None = None,
                  track: bool = False) -> list:
    """Run YOLO on either the rectified board or the original frame.

    When *board_img* is provided (board quad was found), detection runs on the
    perspective-corrected image only.  Each chess square is a consistent size
    and the tilt is removed, so detection accuracy is much higher.

    When *board_img* is None, detection falls back to the original frame.

    When *track* is True, uses model.track() (ByteTrack) instead of
    model.predict() so each detection carries a persistent track_id across
    frames — used by the live webcam mode for object tracking.

    Returns a list of dicts:
        label       — class name  (e.g. "white_pawn" or "white-pawn")
        conf        — confidence score 0-1
        box         — [x1, y1, x2, y2] in pixels
        board_space — True when box coordinates are in the rectified board image
        track_id    — int tracker ID (set when track=True, else None)
    """
    def _run(img, board_space: bool) -> list:
        enhanced     = _clahe_color(img)
        # Compute once per image — adaptive threshold for _verify_piece_color
        board_median = float(np.median(cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)))
        if track:
            try:
                results = model.track(
                    enhanced,
                    conf=config.CONFIDENCE,
                    iou=config.IOU,
                    imgsz=config.INPUT_SIZE,
                    verbose=False,
                    persist=True,   # keep IDs consistent across consecutive calls
                )
            except Exception:
                # ByteTrack unavailable or first-call failure — fall back to predict
                results = model.predict(
                    enhanced,
                    conf=config.CONFIDENCE,
                    iou=config.IOU,
                    imgsz=config.INPUT_SIZE,
                    verbose=False,
                )
        else:
            results = model.predict(
                enhanced,
                conf=config.CONFIDENCE,
                iou=config.IOU,
                imgsz=config.INPUT_SIZE,
                verbose=False,
            )
        dets = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                raw_box   = [int(v) for v in box.xyxy[0].tolist()]
                raw_label = model.names[int(box.cls[0])]
                label     = _verify_piece_color(enhanced, raw_box, raw_label, board_median)
                tid = int(box.id[0]) if (track and box.id is not None) else None
                dets.append({
                    "label":       label,
                    "conf":        float(box.conf[0]),
                    "box":         raw_box,
                    "board_space": board_space,
                    "track_id":    tid,
                })
        return dets

    if board_img is not None:
        return _run(board_img, board_space=True)
    return _run(frame, board_space=False)


# ── 5. Create board state ─────────────────────────────────────────────────────

def create_board_state(detections: list, frame: np.ndarray, quad) -> list:
    """Map piece bounding boxes onto an 8×8 grid.

    Detections tagged board_space=True come from the rectified board image
    and are mapped directly.  Frame-space detections use the perspective
    transform when a board quad is available, otherwise a proportional split.

    When two detections land on the same square the higher-confidence one wins.

    Returns an 8×8 list of FEN characters (empty string = empty square).
    """
    board      = [["" for _ in range(8)] for _ in range(8)]
    board_conf = [[0.0  for _ in range(8)] for _ in range(8)]
    h, w       = frame.shape[:2]
    bsize      = config.BOARD_SIZE   # pixel size of rectified board image

    if quad is not None:
        target = np.array([[0, 0], [8, 0], [8, 8], [0, 8]], dtype="float32")
        M = cv2.getPerspectiveTransform(quad.astype("float32"), target)
    else:
        M = None

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        cx = (x1 + x2) / 2.0
        cy = y2 - (y2 - y1) * 0.1   # bottom-centre anchor

        if det.get("board_space"):
            # Already in rectified-board pixel coordinates — map directly
            col = int(cx * 8 / bsize)
            row = int(cy * 8 / bsize)
        elif M is not None:
            pt  = np.array([[[cx, cy]]], dtype="float32")
            bc  = cv2.perspectiveTransform(pt, M)[0][0]
            col = int(bc[0])
            row = int(bc[1])
        else:
            col = int(cx * 8 / w)
            row = int(cy * 8 / h)

        # Clamp to valid range — bounding boxes at board edges can produce
        # coordinates of 8 or -1 (pixel overflow by 1-2 px); clamping ensures
        # the piece lands on the edge square rather than being silently dropped.
        # Detections far outside the board are already suppressed by YOLO NMS.
        col = max(0, min(7, col))
        row = max(0, min(7, row))
        if det["conf"] >= board_conf[row][col]:
            board[row][col]      = label_to_fen(det["label"])
            board_conf[row][col] = det["conf"]

    return board


def _backproject_boxes(board_dets: list, quad: np.ndarray) -> list:
    """Map detection boxes from rectified-board pixel space back to frame space.

    Used so the annotated_frame view still shows labelled bounding boxes even
    when detection was run on the perspective-corrected board.
    """
    bsize = config.BOARD_SIZE
    src_corners = np.array(
        [[0, 0], [bsize - 1, 0], [bsize - 1, bsize - 1], [0, bsize - 1]],
        dtype="float32",
    )
    M_inv = cv2.getPerspectiveTransform(src_corners, quad)

    projected = []
    for det in board_dets:
        x1, y1, x2, y2 = det["box"]
        corners = np.array(
            [[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]], dtype="float32"
        )
        pc = cv2.perspectiveTransform(corners, M_inv)[:, 0, :]
        projected.append({
            **det,
            "box":         [int(pc[:, 0].min()), int(pc[:, 1].min()),
                            int(pc[:, 0].max()), int(pc[:, 1].max())],
            "board_space": False,
        })
    return projected


# ── 6. Draw results ───────────────────────────────────────────────────────────
def _deduplicate_board_dets(board_dets: list) -> list:
    """Keep only the highest-confidence detection per board grid square.

    Mirrors the winner-takes-all logic in create_board_state so the annotated
    overlay is consistent with the board state.  Without this, two competing
    detections at the same square (e.g. 'black-pawn' 0.95 and 'white-pawn'
    0.97) both appear in the overlay even though only the higher-confidence
    one ends up in the board state — creating a confusing mismatch where
    the overlay label disagrees with the FEN character shown for that square.
    """
    bsize = config.BOARD_SIZE
    best: dict = {}
    for det in board_dets:
        x1, y1, x2, y2 = det["box"]
        cx  = (x1 + x2) / 2.0
        cy  = y2 - (y2 - y1) * 0.1
        col = max(0, min(7, int(cx * 8 / bsize)))
        row = max(0, min(7, int(cy * 8 / bsize)))
        key = (row, col)
        if key not in best or det["conf"] > best[key]["conf"]:
            best[key] = det
    return list(best.values())

def draw_results(frame: np.ndarray, quad, board_img: np.ndarray,
                 detections: list) -> tuple[np.ndarray, np.ndarray]:
    """Produce two annotated images.

    annotated_frame — original image with green board outline + piece boxes.
    annotated_board — perspective-corrected board with 8×8 grid overlay.

    When detection ran on the rectified board (board_space=True) the boxes are
    back-projected to frame coordinates so the main view still shows labels.
    """
    annotated_frame = draw_board_outline(frame, quad) if quad is not None else frame.copy()

    frame_dets = [d for d in detections if not d.get("board_space")]
    board_dets = [d for d in detections if d.get("board_space")]
    # Deduplicate: only show the winner per square so overlay matches board state
    if board_dets:
        board_dets = _deduplicate_board_dets(board_dets)

    if board_dets and quad is not None:
        # Back-project to frame coords so labels appear on the original image too
        annotated_frame = draw_detections(annotated_frame, _backproject_boxes(board_dets, quad))
    elif frame_dets:
        annotated_frame = draw_detections(annotated_frame, frame_dets)

    annotated_board = draw_grid(board_img)
    annotated_board = draw_detections(annotated_board, board_dets)

    return annotated_frame, annotated_board


# ── 7. Motion detection ───────────────────────────────────────────────────────

def detect_motion(frame: np.ndarray, bg_subtractor) -> np.ndarray:
    """Apply MOG2 background subtraction to detect moving pieces or hands.

    Removes shadow pixels (MOG2 marks them as 127 in the foreground mask),
    applies morphological closing to fill holes in piece silhouettes, and
    returns a binary mask (uint8) where 255 = foreground motion detected.

    *bg_subtractor* must be a cv2.BackgroundSubtractorMOG2 instance that
    is persisted across frames (e.g. via st.session_state in the live
    webcam mode) so the background model builds up over time.
    """
    fg_mask = bg_subtractor.apply(frame)
    # MOG2 marks shadows as 127 — retain only definite foreground
    _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
    kernel     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_mask    = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return fg_mask
