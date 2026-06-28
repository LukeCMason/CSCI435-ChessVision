"""
train.py — Fine-tune the chess-piece YOLO model using transfer learning.

Two complementary datasets are used:
  1. Chess Detection  (../data/Chess Detection/)
       81 real photographs with Pascal VOC XML bounding-box annotations.
  2. Chess Positions  (../data/Chess Positions/train/)
       300 digital board renders.  The FEN string encoded in each filename
       describes exactly which piece sits on every square, so YOLO labels
       are generated automatically — no separate annotation file is needed.

Run with:
    python train.py

Output: ../models/best_finetuned.pt
"""

import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

import cv2
import yaml
from ultralytics import YOLO

import config


# ── Preprocessing helper ──────────────────────────────────────────────────────

def _clahe_color(img: np.ndarray) -> np.ndarray:
    """Apply CLAHE via the LAB L-channel to normalise contrast while preserving
    hue and saturation.  Applied to every training image so the model learns
    from contrast-normalised inputs — the same preprocessing used at inference.

    Grayscale conversion is intentionally avoided: colour is the primary
    discriminator between black and white chess pieces.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

# ── Class definitions ─────────────────────────────────────────────────────────
# Must match the pre-trained model's class order.
CLASSES = [
    "black-bishop", "black-king", "black-knight",
    "black-pawn",   "black-queen", "black-rook",
    "white-bishop", "white-king", "white-knight",
    "white-pawn",   "white-queen", "white-rook",
]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}

# FEN character → class ID (used for Chess Positions FEN-from-filename)
_FEN_TO_CLASS = {
    'b': CLASS_TO_ID['black-bishop'], 'k': CLASS_TO_ID['black-king'],
    'n': CLASS_TO_ID['black-knight'], 'p': CLASS_TO_ID['black-pawn'],
    'q': CLASS_TO_ID['black-queen'],  'r': CLASS_TO_ID['black-rook'],
    'B': CLASS_TO_ID['white-bishop'], 'K': CLASS_TO_ID['white-king'],
    'N': CLASS_TO_ID['white-knight'], 'P': CLASS_TO_ID['white-pawn'],
    'Q': CLASS_TO_ID['white-queen'],  'R': CLASS_TO_ID['white-rook'],
}


def _norm(name: str) -> str:
    return name.strip().lower().replace("_", "-")


# ── Dataset builder 1: Chess Detection (Pascal VOC XML) ───────────────────────

def parse_voc_xml(xml_path: Path, img_w: int, img_h: int) -> list[str]:
    """Parse a Pascal VOC XML file and return YOLO-format annotation lines."""
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError:
        return []

    lines = []
    for obj in tree.findall("object"):
        raw_name = obj.findtext("name", default="").strip()
        norm_name = _norm(raw_name)
        if norm_name not in CLASS_TO_ID:
            print(f"  [skip] Unknown class '{raw_name}' in {xml_path.name}")
            continue

        bnd = obj.find("bndbox")
        xmin = float(bnd.findtext("xmin"))
        ymin = float(bnd.findtext("ymin"))
        xmax = float(bnd.findtext("xmax"))
        ymax = float(bnd.findtext("ymax"))

        cx = ((xmin + xmax) / 2) / img_w
        cy = ((ymin + ymax) / 2) / img_h
        bw = (xmax - xmin) / img_w
        bh = (ymax - ymin) / img_h

        cls_id = CLASS_TO_ID[norm_name]
        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    return lines


def build_detection_dataset(src_images: Path, src_annotations: Path,
                             out_dir: Path, val_ratio: float = 0.15) -> int:
    """Convert Chess Detection Pascal VOC dataset to YOLO directory layout."""
    img_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG"}

    images = sorted(
        p for p in src_images.iterdir()
        if p.suffix in img_extensions
    )
    print(f"  Found {len(images)} images in {src_images.name}")

    split = max(1, int(len(images) * (1 - val_ratio)))
    train_imgs = images[:split]
    val_imgs   = images[split:]

    converted = 0
    for split_name, split_imgs in [("train", train_imgs), ("val", val_imgs)]:
        img_out = out_dir / split_name / "images"
        lbl_out = out_dir / split_name / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_path in split_imgs:
            xml_path = src_annotations / (img_path.stem + ".xml")
            if not xml_path.exists():
                continue

            im = cv2.imread(str(img_path))
            if im is None:
                continue
            h, w = im.shape[:2]

            lines = parse_voc_xml(xml_path, w, h)
            if not lines:
                continue

            cv2.imwrite(str(img_out / img_path.name), _clahe_color(im))
            (lbl_out / (img_path.stem + ".txt")).write_text("\n".join(lines))
            converted += 1

    return converted


# ── Dataset builder 2: Chess Positions (FEN from filename) ────────────────────

def _fen_from_filename(filename: str) -> str:
    """Recover the FEN rank string from an image filename.

    The dataset encodes the FEN as the filename stem with '/' replaced by '-'.
    Example: '1b1B1b2-2pK2q1-...' → '1b1B1b2/2pK2q1/...'
    """
    return Path(filename).stem.replace("-", "/")


def _fen_to_yolo(fen: str) -> list[str]:
    """Generate YOLO annotation lines from a FEN string.

    Assumes the board fills the entire image (400×400 px).
    Each square is annotated as a 1/8 × 1/8 normalised bounding box.
    """
    ranks = fen.split("/")
    if len(ranks) != 8:
        return []

    lines = []
    for row, rank in enumerate(ranks):
        col = 0
        for ch in rank:
            if ch.isdigit():
                col += int(ch)
            elif ch in _FEN_TO_CLASS:
                cls_id = _FEN_TO_CLASS[ch]
                cx = (col + 0.5) / 8.0
                cy = (row + 0.5) / 8.0
                lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {0.125:.6f} {0.125:.6f}")
                col += 1
            else:
                col += 1   # unknown char; skip

    return lines


def add_positions_dataset(src_dir: Path, out_dir: Path,
                           n_images: int = 300, val_ratio: float = 0.15) -> int:
    """Sample *n_images* from Chess Positions and add them to the YOLO dataset.

    Files are read by sorted name order for reproducibility; only the first
    *n_images* are used so we never have to enumerate the full 80 000-file
    directory — we stop as soon as we have enough.
    """
    img_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    # Collect only what we need — stop early to avoid iterating 80k files
    collected: list[Path] = []
    for p in sorted(src_dir.iterdir()):
        if p.suffix.lower() in img_extensions:
            collected.append(p)
        if len(collected) >= n_images:
            break

    print(f"  Sampled {len(collected)} images from {src_dir.name}")

    split = max(1, int(len(collected) * (1 - val_ratio)))
    train_imgs = collected[:split]
    val_imgs   = collected[split:]

    converted = 0
    for split_name, split_imgs in [("train", train_imgs), ("val", val_imgs)]:
        img_out = out_dir / split_name / "images"
        lbl_out = out_dir / split_name / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_path in split_imgs:
            fen   = _fen_from_filename(img_path.name)
            lines = _fen_to_yolo(fen)
            if not lines:
                print(f"  [skip] Could not parse FEN from {img_path.name[:50]}")
                continue

            im = cv2.imread(str(img_path))
            if im is None:
                continue
            cv2.imwrite(str(img_out / img_path.name), _clahe_color(im))
            (lbl_out / (img_path.stem + ".txt")).write_text("\n".join(lines))
            converted += 1

    return converted


# ── YAML helper ───────────────────────────────────────────────────────────────

def write_data_yaml(out_dir: Path) -> Path:
    data = {
        "path":  str(out_dir),
        "train": "train/images",
        "val":   "val/images",
        "nc":    len(CLASSES),
        "names": CLASSES,
    }
    yaml_path = out_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)
    return yaml_path


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    _here = Path(__file__).parent
    raw_root   = _here / ".." / "data"
    out_dir    = _here / ".." / "training_data"
    out_model  = _here / ".." / "models" / "best_finetuned.pt"

    # ── 1. Build dataset ──────────────────────────────────────────────────────
    print("=" * 60)
    print("Building combined YOLO training dataset …")
    print("=" * 60)

    if out_dir.exists():
        shutil.rmtree(out_dir)

    total = 0

    # 1a. Chess Detection (real photos, VOC XML annotations)
    det_images = raw_root / "Chess Detection" / "images"
    det_annots = raw_root / "Chess Detection" / "annotations"  # Pascal VOC XML
    if det_images.exists():
        print("\n[1/2] Chess Detection (real photos + VOC XML)")
        n = build_detection_dataset(det_images, det_annots, out_dir)
        print(f"      → {n} images converted")
        total += n
    else:
        print(f"[1/2] Chess Detection not found — skipping ({det_images})")

    # 1b. Chess Positions (digital renders, FEN from filename)
    pos_train = raw_root / "Chess Positions" / "train"
    if pos_train.exists():
        print("\n[2/2] Chess Positions (digital renders, FEN annotations) — 300 samples")
        n = add_positions_dataset(pos_train, out_dir, n_images=300)
        print(f"      → {n} images converted")
        total += n
    else:
        print(f"[2/2] Chess Positions not found — skipping ({pos_train})")

    print(f"\nDataset total: {total} images")

    if total == 0:
        print("No images to train on. Aborting.")
        return

    yaml_path = write_data_yaml(out_dir)
    print(f"data.yaml written → {yaml_path}")

    # Count images per split for the user
    for split in ("train", "val"):
        d = out_dir / split / "images"
        if d.exists():
            print(f"  {split}: {len(list(d.iterdir()))} images")

    # ── 2. Load base model ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Loading base model …")
    base_model_path = config.TRAIN_BASE_PATH
    if not os.path.exists(base_model_path):
        print(f"ERROR: Base model not found at {base_model_path}")
        return
    print(f"  {base_model_path}")

    model = YOLO(base_model_path)
    model_classes = [model.names[i] for i in range(len(model.names))]
    norm_model = [_norm(c) for c in model_classes]
    norm_ours  = [_norm(c) for c in CLASSES]
    if norm_model != norm_ours:
        print("  NOTE: model class names differ from dataset — detection head will be replaced.")

    # ── 3. Fine-tune ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Starting transfer learning …")
    print("  epochs=20  imgsz=640  batch=8  freeze=10 backbone layers")

    results = model.train(
        data=str(yaml_path),
        epochs=20,
        imgsz=640,
        batch=8,
        freeze=10,        # freeze first 10 layers (backbone); only head adapts
        patience=10,
        lr0=0.001,
        lrf=0.01,
        hsv_v=0.4,        # brightness jitter — key robustness for variable lighting
        hsv_s=0.2,        # saturation jitter — white-balance / camera variation
        project=str(_here / ".." / "runs"),
        name="chess_finetune",
        exist_ok=True,
        verbose=True,
    )

    # ── 4. Save best weights ──────────────────────────────────────────────────
    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    if best_pt.exists():
        shutil.copy(best_pt, out_model)
        print(f"\nFine-tuned model saved → {out_model.resolve()}")
    else:
        print(f"WARNING: best.pt not found at {best_pt}")

    print("\nDone.")


if __name__ == "__main__":
    main()

