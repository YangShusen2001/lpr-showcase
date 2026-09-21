"""Synthetic Chinese license-plate recognition dataset generator.

Renders plates with real Chinese fonts, applies photographic augmentation
(perspective / lighting / blur / noise), then emits grayscale 32x128 crops
plus their CTC labels. 100% rule-generated -> zero annotation error, no
privacy exposure (no real plate photos anywhere).

Output: _dataset/rec/{train,val}_{images.npy,labels.json}
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "_dataset" / "rec"

PROVINCES = "京津冀晋蒙辽吉黑沪苏浙皖闽赣鲁豫鄂湘粤桂琼渝川贵云藏陕甘青宁新"
LETTERS = "ABCDEFGHJKLMNPQRSTUVWXYZ"   # no I / O  (CCPD convention)
DIGITS = "0123456789"
CHARSET = PROVINCES + LETTERS + DIGITS          # 65 symbols
BLANK = len(CHARSET)                             # CTC blank index = 65
NUM_CLASSES = len(CHARSET) + 1                   # 66

FONT_CANDIDATES = [
    "C:/Windows/Fonts/msyhbd.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/msyh.ttc",
]

# plate style -> (body colour, glyph colour, gradient?)
STYLES = {
    "blue":   ((0, 87, 200), (255, 255, 255), False),
    "yellow": ((255, 199, 0), (17, 17, 17), False),
    "green":  ((26, 138, 74), (17, 17, 17), True),
    "white":  ((248, 248, 248), (17, 17, 17), False),
}

PLATE_W, PLATE_H = 440, 140
OUT_W, OUT_H = 128, 32


def load_font(size: int) -> ImageFont.FreeTypeFont:
    for p in FONT_CANDIDATES:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except OSError:
                continue
    raise SystemExit("no Chinese font found")


def random_text(rng: random.Random) -> str:
    prov = rng.choice(PROVINCES)
    letter = rng.choice(LETTERS)
    # ~30% new-energy style: 8 characters total
    n = 6 if rng.random() < 0.30 else 5
    tail = "".join(rng.choice(DIGITS + LETTERS) for _ in range(n))
    return prov + letter + tail


def render_plate(text: str, style: str, rng: random.Random) -> Image.Image:
    body, glyph, grad = STYLES[style]
    img = Image.new("RGB", (PLATE_W, PLATE_H), body)
    d = ImageDraw.Draw(img)

    if grad:  # new-energy green: vertical gradient
        for y in range(PLATE_H):
            t = y / PLATE_H
            col = tuple(int(body[i] * (1 - 0.35 * t) + 255 * 0.35 * t * 0.0) for i in range(3))
            col = (max(0, col[0] - int(20 * t)), min(255, col[1] + int(26 * t)), max(0, col[2] - int(10 * t)))
            d.line((0, y, PLATE_W, y), fill=col)

    d.rounded_rectangle((0, 0, PLATE_W - 1, PLATE_H - 1), radius=14, outline=glyph, width=3)

    font = load_font(78)
    dot_font = load_font(78)
    widths = [d.textlength(c, font=font) for c in text]
    dot_w = d.textlength("·", font=dot_font)
    gap = 5
    total = sum(widths) + gap * (len(text) - 1) + dot_w + gap
    x = (PLATE_W - total) / 2
    y = (PLATE_H - 86) / 2 - 4
    for i, c in enumerate(text):
        d.text((x, y), c, font=font, fill=glyph)
        x += widths[i] + gap
        if i == 1:  # separator dot after the letter
            d.text((x, y - 2), "·", font=dot_font, fill=glyph)
            x += dot_w + gap
    return img


def augment(plate: Image.Image, rng: random.Random) -> Image.Image:
    """Photographic degradation + mild perspective, then grayscale resize."""
    w, h = plate.size
    src = [(0, 0), (w, 0), (w, h), (0, h)]
    j = 0.05
    dst = [(x + rng.uniform(-j, j) * w, y + rng.uniform(-j, j) * h) for x, y in src]
    # PIL QUAD transform maps dst->src, so build the inverse by solving for coefficients
    coeffs = _find_coeffs(dst, src)
    plate = plate.transform((w, h), Image.Transform.PERSPECTIVE, coeffs, Image.Resampling.BICUBIC)

    # lighting gradient
    grad = Image.new("L", (w, h), 0)
    gd = ImageDraw.Draw(grad)
    ang = rng.uniform(0, 6.28)
    dx, dy = int(90 * np.cos(ang)), int(90 * np.sin(ang))
    gd.line((dx, dy, w + dx, h + dy), fill=255, width=int(w * 0.9))
    grad = grad.filter(ImageFilter.GaussianBlur(w * 0.25))
    dark = Image.new("RGB", (w, h), (0, 0, 0))
    plate = Image.composite(plate, dark, grad.point(lambda v: 128 + int(v * 0.5)))

    if rng.random() < 0.7:
        plate = plate.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.6)))
    if rng.random() < 0.5:
        arr = np.asarray(plate).astype(np.int16)
        arr += np.random.normal(0, rng.uniform(3, 14), arr.shape).astype(np.int16)
        plate = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if rng.random() < 0.6:
        f = rng.uniform(0.78, 1.22)
        plate = Image.fromarray(np.clip(np.asarray(plate).astype(np.float32) * f, 0, 255).astype(np.uint8))

    return plate.convert("L").resize((OUT_W, OUT_H), Image.Resampling.BILINEAR)


def _find_coeffs(dst, src):
    """Coefficients for PIL's PERSPECTIVE transform (maps dst quad -> src quad)."""
    a = []
    b = []
    for (dx, dy), (sx, sy) in zip(dst, src):
        a.append([dx, dy, 1, 0, 0, 0, -sx * dx, -sx * dy])
        a.append([0, 0, 0, dx, dy, 1, -sy * dx, -sy * dy])
        b.append(sx)
        b.append(sy)
    return np.linalg.solve(np.array(a, dtype=np.float64), np.array(b, dtype=np.float64)).tolist()


def build(n: int, seed: int):
    rng = random.Random(seed)
    images = np.zeros((n, OUT_H, OUT_W), dtype=np.uint8)
    labels, lengths = [], []
    styles = list(STYLES)
    for i in range(n):
        text = random_text(rng)
        style = rng.choice(styles)
        plate = render_plate(text, style, rng)
        images[i] = np.asarray(augment(plate, rng))
        labels.append([CHARSET.index(c) for c in text])
        lengths.append(len(text))
    return images, labels, lengths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=60000)
    ap.add_argument("--val", type=int, default=6000)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    for split, n, seed in (("train", args.train, 20260917), ("val", args.val, 777)):
        images, labels, lengths = build(n, seed)
        np.save(OUT / f"{split}_images.npy", images)
        (OUT / f"{split}_labels.json").write_text(
            json.dumps({"charset": CHARSET, "labels": labels, "lengths": lengths}, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"{split}: {images.shape} saved")


if __name__ == "__main__":
    main()
