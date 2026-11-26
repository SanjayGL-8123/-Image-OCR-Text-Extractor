"""
Read all the text inside every image in a folder and save it into ONE text file.
Supports resume: images already in the output file are skipped automatically.

Supported Engines:
- Windows Native OCR (winocr) - Fast and built-in (preferred)
- EasyOCR - PyTorch based (installed, lazy-loaded only if winocr fails)
- Tesseract OCR (pytesseract) - If installed
"""

import os
import sys
from PIL import Image

# ---------- SETTINGS ----------
IMAGE_FOLDERS = [
    r"C:\Users\\Music\photo\NOTRS",
]
OUTPUT_FILE = r"C:\Users\\Music\photo\all_image_text.txt"
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Leave empty [] to save the text of ALL images.
# Or add words to keep only matching images, e.g. ["pwn", "ui/ux", "penpot"]
KEYWORDS = []
# ------------------------------

EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff")

# Detect available OCR engines (easyocr imported lazily below)
HAS_WINOCR = False
HAS_EASYOCR = False
HAS_PYTESSERACT = False
_easyocr_reader = None

try:
    import winocr
    HAS_WINOCR = True
except ImportError:
    pass

# Check easyocr availability without importing (avoid slow torch load at startup)
try:
    import importlib.util
    if importlib.util.find_spec("easyocr") is not None:
        HAS_EASYOCR = True
except Exception:
    pass

try:
    import pytesseract
    if os.path.exists(TESSERACT_PATH):
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
        HAS_PYTESSERACT = True
except ImportError:
    pass


def get_easyocr_reader():
    """Lazily initialize easyocr reader on first use."""
    global _easyocr_reader
    if _easyocr_reader is None:
        import easyocr
        print("  [Loading EasyOCR model, this may take a moment...]")
        _easyocr_reader = easyocr.Reader(["en"], gpu=False)
    return _easyocr_reader


def ocr_image(path):
    """Extract text from an image using the best available OCR engine."""
    # 1. Native Windows OCR (Fastest & most accurate for screen/notes)
    if HAS_WINOCR:
        try:
            with Image.open(path) as img:
                res = winocr.recognize_pil_sync(img)
                return res.get("text", "").strip()
        except Exception:
            pass

    # 2. EasyOCR (lazy load — only imports torch when actually needed)
    if HAS_EASYOCR:
        try:
            reader = get_easyocr_reader()
            res = reader.readtext(path, detail=0)
            return "\n".join(res).strip()
        except Exception:
            pass

    # 3. Pytesseract
    if HAS_PYTESSERACT:
        try:
            with Image.open(path) as img:
                return pytesseract.image_to_string(img).strip()
        except Exception:
            pass

    return ""


def find_images(folders):
    """Return every image path in the folders and their subfolders (no duplicates)."""
    paths = set()
    for folder in folders:
        if not os.path.isdir(folder):
            print(f"WARNING: folder not found: {folder}")
            continue
        for root, _, files in os.walk(folder):
            for name in files:
                if name.lower().endswith(EXTENSIONS):
                    paths.add(os.path.normcase(os.path.abspath(os.path.join(root, name))))
    return sorted(paths)


def load_already_done(output_file):
    """Read the output file and return a set of already-processed basenames."""
    done = set()
    if not os.path.exists(output_file):
        return done
    with open(output_file, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line.startswith("=====") and line.endswith("====="):
                # e.g. "===== myimage.jpg =====" -> "myimage.jpg"
                name = line[5:-5].strip()
                done.add(os.path.normcase(name))
    return done


def main():
    print("Initializing OCR engine...")
    if HAS_WINOCR:
        print("Using: Windows Native OCR (winocr) - Fast mode")
    elif HAS_EASYOCR:
        print("Using: EasyOCR (slow on CPU, ~5s/image)")
    elif HAS_PYTESSERACT:
        print("Using: Tesseract OCR")
    else:
        print("ERROR: No OCR engine available. Please run: python -m pip install winocr")
        sys.exit(1)

    images = find_images(IMAGE_FOLDERS)
    total = len(images)
    print(f"Found {total} images in {len(IMAGE_FOLDERS)} folder(s)")

    # Resume: skip images already in the output file
    done_set = load_already_done(OUTPUT_FILE)
    if done_set:
        print(f"Resuming: {len(done_set)} images already done, skipping them.")

    pending = [p for p in images if os.path.normcase(os.path.basename(p)) not in done_set]
    print(f"Remaining: {len(pending)} images to process")

    if not pending:
        print("All images already processed!")
        return

    saved = 0
    # Append to existing file
    with open(OUTPUT_FILE, "a", encoding="utf-8") as out:
        for i, path in enumerate(pending, start=len(done_set) + 1):
            try:
                text = ocr_image(path)
            except Exception as e:
                print(f"[{i}/{total}] ERROR {os.path.basename(path)}: {e}")
                continue

            # Keyword filter (skipped when KEYWORDS is empty)
            if KEYWORDS and not any(k.lower() in text.lower() for k in KEYWORDS):
                print(f"[{i}/{total}] skipped {os.path.basename(path)}")
                continue

            out.write(f"===== {os.path.basename(path)} =====\n{text}\n\n")
            out.flush()  # progress is saved even if you stop the script
            saved += 1
            print(f"[{i}/{total}] done {os.path.basename(path)}")

    print(f"\nFinished. {saved} new images saved to {OUTPUT_FILE}")
    print(f"Total in file: {len(done_set) + saved}")


if __name__ == "__main__":
    main()
