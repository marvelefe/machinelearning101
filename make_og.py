#!/usr/bin/env python3
"""Generate public/og-image.png (1200x630) for machinelearning101.

Brand: blue-600 (#2563eb) -> indigo-700 (#4338ca) gradient (matches the
from-blue-600 to-indigo-700 CTA section), white text, ML-101 motif.
Placed exactly where composables/useSEO.ts DEFAULT_SEO points:
https://machinelearning101.com/og-image.png
"""
from PIL import Image, ImageDraw, ImageFont

W, H = 1200, 630
C1 = (37, 99, 235)    # blue-600
C2 = (67, 56, 202)    # indigo-700
WHITE = (255, 255, 255)
SOFT = (199, 210, 254)  # indigo-200

img = Image.new("RGB", (W, H))
d = ImageDraw.Draw(img)

# vertical gradient
for y in range(H):
    t = y / (H - 1)
    r = int(C1[0] + (C2[0] - C1[0]) * t)
    g = int(C1[1] + (C2[1] - C1[1]) * t)
    b = int(C1[2] + (C2[2] - C1[2]) * t)
    d.line([(0, y), (W, y)], fill=(r, g, b))

# faint neural-network motif: nodes + edges (bottom right)
import random
random.seed(42)
nodes = [(x, y) for x in range(780, 1160, 95) for y in range(430, 600, 55)]
for i, (x1, y1) in enumerate(nodes):
    for j in range(i + 1, len(nodes)):
        x2, y2 = nodes[j]
        if abs(x2 - x1) < 220 and abs(y2 - y1) < 130:
            d.line([(x1, y1), (x2, y2)], fill=(199, 210, 254, 60) if False else (139, 159, 235), width=2)
for x, y in nodes:
    d.ellipse([x - 9, y - 9, x + 9, y + 9], fill=(224, 231, 255), outline=(255, 255, 255), width=2)

def font(path, size):
    return ImageFont.truetype(path, size)

title_font = font("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 64)
sub_font = font("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 36)
tiny_font = font("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)

# badge-like "ML 101" chip
chip = "ML 101"
bb = d.textbbox((0, 0), chip, font=tiny_font)
cw = bb[2] - bb[0] + 48
d.rounded_rectangle([(W // 2 - cw // 2, 105), (W // 2 + cw // 2, 165)], radius=30, fill=(255, 255, 255), outline=None)
d.text((W // 2 - (bb[2] - bb[0]) // 2 - bb[0] + 24, 118), chip, font=tiny_font, fill=C2)

def center_text(text, fnt, fill, y):
    bbox = d.textbbox((0, 0), text, font=fnt)
    tw = bbox[2] - bbox[0]
    d.text(((W - tw) // 2 - bbox[0], y), text, font=fnt, fill=fill)

center_text("Machine Learning 101", title_font, WHITE, 210)
center_text("Learn machine learning by doing", sub_font, SOFT, 315)
center_text("— right in your browser —", sub_font, SOFT, 372)
center_text("zero setup \u00b7 visual-first \u00b7 open source", tiny_font, (226, 232, 253), 480)

img.save("/home/paperclip/work-ml101/public/og-image.png", "PNG")
print("saved ml101 og-image.png")