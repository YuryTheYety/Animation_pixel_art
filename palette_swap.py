"""Genere des variantes de couleur d'un personnage "sans features" par
transformation HSV (teinte + saturation remplacees, la luminosite -- donc
tout l'ombrage/la forme -- est preservee). Utilise pour transformer un seul
personnage dessine a la main en plusieurs "identites" distinctes,
necessaires pour le fine-tuning (le modele a besoin de plusieurs identites
pour apprendre a separer pose et apparence, cf. discussion).

Usage:
    python palette_swap.py --source /chemin/vers/mannequin --out /chemin/vers/variants
"""
import argparse
import colorsys
import os
import shutil

import numpy as np
from PIL import Image

# (nom, teinte cible en degres 0-360, multiplicateur de saturation, multiplicateur de luminosite)
PALETTES = [
    ("hale_bronze",       30,  1.2, 0.85),
    ("brun_fonce",        20,  1.4, 0.55),
    ("olive_verdatre",    70,  0.9, 1.00),
    ("ambre_dore",        45,  1.1, 1.05),
    ("gris_pierre",        0,  0.05, 0.75),
    ("vert_zombie",       120, 1.3, 0.90),
    ("bleu_glace",        200, 0.9, 1.00),
    ("violet_lavande",    270, 0.8, 0.95),
    ("rouge_brique",        5, 1.5, 0.80),
    ("noir_anthracite",     0, 0.10, 0.25),
    ("blanc_albatre",       0, 0.05, 1.15),
    ("marron_fourrure",    25, 0.9, 0.60),
    ("vert_fonce",         140, 1.1, 0.50),
    ("bleu_gris_ardoise",  190, 1.2, 0.70),
]


def hsv_recolor(img, target_hue_deg, sat_scale, val_scale):
    """img: PIL RGBA. Remplace H et S (mises a l'echelle) par la teinte
    cible, ajuste V multiplicativement, alpha inchange."""
    arr = np.asarray(img.convert("RGBA"), dtype=np.float32) / 255.0
    r, g, b, a = arr[..., 0], arr[..., 1], arr[..., 2], arr[..., 3]

    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    v = maxc
    s = np.where(maxc > 0, (maxc - minc) / np.where(maxc > 0, maxc, 1), 0.0)

    target_h = (target_hue_deg % 360) / 360.0
    new_s = np.clip(s * sat_scale, 0.0, 1.0)
    new_v = np.clip(v * val_scale, 0.0, 1.0)

    h_arr = np.full_like(v, target_h)
    rgb_new = np.stack([
        _hsv_to_rgb_vec(h_arr, new_s, new_v, channel=c) for c in range(3)
    ], axis=-1)

    out = np.stack([rgb_new[..., 0], rgb_new[..., 1], rgb_new[..., 2], a], axis=-1)
    out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="RGBA")


def _hsv_to_rgb_vec(h, s, v, channel):
    # Vectorized HSV->RGB (standard formula), returns one channel at a time.
    i = np.floor(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i = i.astype(int) % 6

    r = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5], [v, q, p, p, t, v])
    g = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5], [t, v, v, q, p, p])
    b = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5], [p, p, t, v, v, q])
    return [r, g, b][channel]


def generate_variants(source_dir, out_dir, palettes=PALETTES):
    png_files = []
    for root, _, files in os.walk(source_dir):
        for f in files:
            if f.lower().endswith(".png"):
                png_files.append(os.path.relpath(os.path.join(root, f), source_dir))

    os.makedirs(out_dir, exist_ok=True)
    for name, hue, sat_scale, val_scale in palettes:
        variant_dir = os.path.join(out_dir, name)
        print(f"Generation variante '{name}' (H={hue}, sat x{sat_scale}, val x{val_scale}) ...")
        for rel_path in png_files:
            src_path = os.path.join(source_dir, rel_path)
            dst_path = os.path.join(variant_dir, rel_path)
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            img = Image.open(src_path)
            recolored = hsv_recolor(img, hue, sat_scale, val_scale)
            recolored.save(dst_path)
    print(f"{len(palettes)} variantes generees dans {out_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Dossier du personnage source (featureless).")
    parser.add_argument("--out", required=True, help="Dossier ou ecrire les variantes recolorees.")
    args = parser.parse_args()
    generate_variants(args.source, args.out)


if __name__ == "__main__":
    main()
