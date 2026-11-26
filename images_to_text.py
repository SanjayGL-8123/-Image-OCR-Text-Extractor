"""
Batch image -> text (layout preserved) using EasyOCR.

Install once:
    pip install easyocr opencv-python numpy

Usage:
    python images_to_text.py  input_folder  output_folder
Optional:
    --crop-top 0.05 --crop-bottom 0.05   (cut phone status bar / buttons, as fraction)
    --gpu                                (if you have an NVIDIA GPU)
"""
import argparse
import statistics
from pathlib import Path

import cv2
import easyocr
import numpy as np

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def run_ocr(reader, img):
    """Return list of (box, text, confidence)."""
    return reader.readtext(img, paragraph=False, decoder="greedy")


def score(results):
    # more confident characters = better reading
    return sum(len(t) * c for _, t, c in results)


def best_ocr(reader, img):
    """Try normal and inverted image (helps with dark backgrounds)."""
    normal = run_ocr(reader, img)
    inverted = run_ocr(reader, 255 - img)
    return normal if score(normal) >= score(inverted) else inverted


def preprocess(path, crop_top, crop_bottom):
    img = cv2.imread(str(path))
    if img is None:
        return None
    h = img.shape[0]
    img = img[int(h * crop_top): int(h * (1 - crop_bottom)), :]
    # upscale small images, OCR likes bigger text
    if img.shape[1] < 1500:
        scale = 1500 / img.shape[1]
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return img


def layout_text(results, min_conf=0.30):
    """Rebuild lines and columns from box positions so the shape stays the same."""
    items = []
    for box, text, conf in results:
        text = text.strip()
        if not text or conf < min_conf:
            continue
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        items.append({
            "x": min(xs), "x2": max(xs),
            "y": (min(ys) + max(ys)) / 2,
            "h": max(ys) - min(ys),
            "text": text,
        })
    if not items:
        return ""

    med_h = statistics.median(i["h"] for i in items)
    char_w = statistics.median((i["x2"] - i["x"]) / max(len(i["text"]), 1) for i in items)
    char_w = max(char_w, 1)

    # group into rows
    items.sort(key=lambda i: i["y"])
    rows, current = [], [items[0]]
    for it in items[1:]:
        row_y = sum(c["y"] for c in current) / len(current)
        if abs(it["y"] - row_y) <= med_h * 0.6:
            current.append(it)
        else:
            rows.append(current)
            current = [it]
    rows.append(current)

    # build each line with spaces so columns line up
    lines, prev_y = [], None
    for row in rows:
        row.sort(key=lambda i: i["x"])
        y = sum(c["y"] for c in row) / len(row)
        if prev_y is not None and (y - prev_y) > med_h * 2.2:
            lines.append("")  # blank line for a bigger vertical gap
        prev_y = y

        line = ""
        for it in row:
            col = int(it["x"] / char_w)
            if col > len(line):
                line += " " * (col - len(line))
            elif line:
                line += " "
            line += it["text"]
        lines.append(line.rstrip())
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_dir")
    ap.add_argument("output_dir")
    ap.add_argument("--langs", nargs="+", default=["en"])
    ap.add_argument("--crop-top", type=float, default=0.0)
    ap.add_argument("--crop-bottom", type=float, default=0.0)
    ap.add_argument("--gpu", action="store_true")
    args = ap.parse_args()

    in_dir, out_dir = Path(args.input_dir), Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in in_dir.iterdir() if p.suffix.lower() in EXTS)
    print(f"Found {len(files)} images")

    reader = easyocr.Reader(args.langs, gpu=args.gpu)
    combined = []

    for n, path in enumerate(files, 1):
        img = preprocess(path, args.crop_top, args.crop_bottom)
        if img is None:
            print(f"[{n}/{len(files)}] cannot read {path.name}")
            continue
        text = layout_text(best_ocr(reader, img))
        (out_dir / f"{path.stem}.txt").write_text(text, encoding="utf-8")
        combined.append(f"===== {path.name} =====\n{text}\n")
        print(f"[{n}/{len(files)}] {path.name}: {len(text)} characters")

    (out_dir / "ALL_TEXT.txt").write_text("\n".join(combined), encoding="utf-8")
    print("Done. See ALL_TEXT.txt and one .txt per image.")


if __name__ == "__main__":
    main()