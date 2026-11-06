"""Convierte las anotaciones de XFUND-ES (formularios reales escaneados en español)
al formato de dataset de reconocimiento que usa PaddleX/PaddleOCR, y las añade
al dataset existente en dataset/ (junto a las líneas sintéticas de hojas de vida).

Cada entidad de XFUND trae una caja (box) + texto exacto. Recortamos esa caja
de la imagen original y la guardamos como una muestra de línea de texto.
"""

import json
import os
import random

from PIL import Image

random.seed(123)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
XFUND_DIR = os.path.join(BASE_DIR, "xfund_es_sample")
XFUND_JSON = os.path.join(XFUND_DIR, "es.val.json")
XFUND_IMAGES_DIR = os.path.join(XFUND_DIR, "images")

DATASET_DIR = os.path.join(BASE_DIR, "dataset")
IMAGES_DIR = os.path.join(DATASET_DIR, "images")

MIN_WIDTH = 8
MIN_HEIGHT = 8
MAX_ASPECT_RATIO = 25  # descarta cajas absurdamente alargadas (probable multi-línea)
VAL_RATIO = 0.15


def load_dict(path):
    with open(path, encoding="utf-8") as f:
        return set(line.rstrip("\n") for line in f)


def is_usable(text, dict_chars):
    text = text.strip()
    if not text:
        return False
    for c in text:
        if c != " " and c not in dict_chars:
            return False
    return True


def collect_samples():
    with open(XFUND_JSON, encoding="utf-8") as f:
        data = json.load(f)

    dict_chars = load_dict(os.path.join(DATASET_DIR, "dict.txt"))

    samples = []  # (image_path, box, text)
    skipped_chars = 0
    skipped_shape = 0

    for doc in data["documents"]:
        img_path = os.path.join(XFUND_IMAGES_DIR, doc["img"]["fname"])
        if not os.path.exists(img_path):
            continue
        for ent in doc["document"]:
            text = ent["text"].strip()
            box = ent["box"]
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1

            if not is_usable(text, dict_chars):
                skipped_chars += 1
                continue
            if w < MIN_WIDTH or h < MIN_HEIGHT:
                skipped_shape += 1
                continue
            if w / max(h, 1) > MAX_ASPECT_RATIO:
                skipped_shape += 1
                continue

            samples.append((img_path, (x1, y1, x2, y2), text))

    print(f"Entidades usables: {len(samples)}")
    print(f"  descartadas por caracteres fuera de dict.txt: {skipped_chars}")
    print(f"  descartadas por forma (muy pequeñas/alargadas): {skipped_shape}")
    return samples


def crop_and_save(samples):
    random.shuffle(samples)
    n_val = max(1, int(len(samples) * VAL_RATIO))
    val_samples = samples[:n_val]
    train_samples = samples[n_val:]

    new_lines = {"train": [], "val": []}
    opened_images = {}

    for split, split_samples in (("train", train_samples), ("val", val_samples)):
        split_dir = os.path.join(IMAGES_DIR, split)
        os.makedirs(split_dir, exist_ok=True)
        for idx, (img_path, box, text) in enumerate(split_samples):
            if img_path not in opened_images:
                opened_images[img_path] = Image.open(img_path).convert("RGB")
            img = opened_images[img_path]
            crop = img.crop(box)

            file_name = f"xfund_es_{split}_{idx:04d}.png"
            crop.save(os.path.join(split_dir, file_name))
            new_lines[split].append(f"images/{split}/{file_name}\t{text}")

    return new_lines


def append_to_labels(new_lines):
    for split in ("train", "val"):
        label_path = os.path.join(DATASET_DIR, f"{split}.txt")
        with open(label_path, "a", encoding="utf-8", newline="\n") as f:
            for line in new_lines[split]:
                f.write(line + "\n")
        print(f"Añadidas {len(new_lines[split])} líneas a {label_path}")


def main():
    samples = collect_samples()
    new_lines = crop_and_save(samples)
    append_to_labels(new_lines)
    print("Listo.")


if __name__ == "__main__":
    main()
