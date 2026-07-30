"""(CPU) Verifie le dataset pretraite : alignement des paires, shapes, et
affiche visuellement quelques paires (reference / pose / cible) pour
verification humaine.

Usage:
    python sanity_check.py [--n 5] [--seed 0]
"""
import argparse
import json
import os
import random

import numpy as np
import torch
from PIL import Image

import config


def load_manifest():
    if not os.path.exists(config.MANIFEST_PATH):
        raise FileNotFoundError(
            f"{config.MANIFEST_PATH} introuvable : lancer data_prep.py d'abord."
        )
    with open(config.MANIFEST_PATH) as f:
        return json.load(f)


def load_pose_maps():
    blob = torch.load(config.POSE_MAPS_PATH, weights_only=False)
    return blob["maps"]  # dict (row,col) -> tensor[1,H,W]


def check_shapes(reference, pose_map, target, frame_size):
    assert reference.shape == (4, frame_size, frame_size), (
        f"reference shape inattendue: {tuple(reference.shape)}"
    )
    assert target.shape == (4, frame_size, frame_size), (
        f"target shape inattendue: {tuple(target.shape)}"
    )
    assert pose_map.shape == (config.POSE_CHANNELS, frame_size, frame_size), (
        f"pose_map shape inattendue: {tuple(pose_map.shape)}"
    )
    for name, t in [("reference", reference), ("target", target), ("pose_map", pose_map)]:
        assert t.dtype == torch.float32, f"{name} dtype inattendu: {t.dtype}"
        assert t.min() >= 0.0 - 1e-5 and t.max() <= 1.0 + 1e-5, (
            f"{name} hors de [0,1]: min={t.min():.3f} max={t.max():.3f}"
        )


def silhouette_iou(pose_map, target, alpha_threshold=0.02):
    """Chevauchement grossier entre le masque de la pose (derivee du
    personnage neutre) et le masque alpha de la cible (meme pose, personnage
    different). Un IoU tres bas est un signal de desalignement des donnees."""
    pose_mask = (pose_map[0] > alpha_threshold).numpy()
    target_mask = (target[3] > alpha_threshold).numpy()
    inter = np.logical_and(pose_mask, target_mask).sum()
    union = np.logical_or(pose_mask, target_mask).sum()
    if union == 0:
        return 1.0
    return float(inter / union)


def tensor_to_pil_rgba(t):
    arr = (t.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGBA")


def pose_tensor_to_pil(t):
    gray = (t[0].numpy() * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(gray, mode="L").convert("RGBA")


def checkerboard(size, cell=8):
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    for y in range(0, size, cell):
        for x in range(0, size, cell):
            if ((x // cell) + (y // cell)) % 2 == 0:
                arr[y:y + cell, x:x + cell] = 200
            else:
                arr[y:y + cell, x:x + cell] = 160
    return Image.fromarray(arr, mode="RGB")


def composite_on_checkerboard(rgba_img, scale=4):
    up = rgba_img.resize(
        (rgba_img.width * scale, rgba_img.height * scale), Image.NEAREST
    )
    bg = checkerboard(up.width).convert("RGBA")
    bg.alpha_composite(up)
    return bg.convert("RGB")


def run(n=5, seed=0):
    manifest = load_manifest()
    frame_size = manifest["frame_size"]
    pose_coords_all = [tuple(c) for c in manifest["pose_coords"]]
    characters = manifest["characters"]
    if not characters:
        raise RuntimeError("Manifest vide, aucun personnage a verifier.")

    pose_maps = load_pose_maps()

    rng = random.Random(seed)
    samples = []
    n = min(n, len(characters))
    chosen_chars = rng.sample(characters, n)
    for char_entry in chosen_chars:
        char_path = os.path.join(config.PREPARED_DIR, char_entry["file"])
        blob = torch.load(char_path, weights_only=False)
        pose_coord = rng.choice(pose_coords_all)
        reference = blob["reference"]
        target = blob["targets"][pose_coord]
        pose_map = pose_maps[pose_coord]

        check_shapes(reference, pose_map, target, frame_size)
        iou = silhouette_iou(pose_map, target)

        samples.append({
            "char_id": char_entry["id"],
            "pose_coord": pose_coord,
            "reference": reference,
            "pose_map": pose_map,
            "target": target,
            "iou": iou,
        })
        print(f"[OK] {char_entry['id']:<20} pose={pose_coord}  "
              f"shapes ok  silhouette_IoU={iou:.2f}")

    low_iou = [s for s in samples if s["iou"] < 0.15]
    if low_iou:
        print(f"\n[!] {len(low_iou)} paire(s) avec un IoU de silhouette tres bas "
              f"(<0.15) : verifier visuellement, possible desalignement.")
    else:
        print(f"\nToutes les paires ont un IoU de silhouette raisonnable (>=0.15).")

    scale = 4
    pad = 6
    cell = frame_size * scale
    cols = 3  # reference | pose | cible
    labels = ["reference", "pose", "cible"]
    header_h = 20
    W = cols * cell + (cols + 1) * pad
    H = header_h + n * cell + (n + 1) * pad
    canvas = Image.new("RGB", (W, H), (30, 30, 30))

    try:
        from PIL import ImageDraw
        draw = ImageDraw.Draw(canvas)
        for j, label in enumerate(labels):
            x = pad + j * (cell + pad)
            draw.text((x, 2), label, fill=(255, 255, 255))
    except Exception:
        pass

    for i, s in enumerate(samples):
        y = header_h + pad + i * (cell + pad)
        imgs = [
            composite_on_checkerboard(tensor_to_pil_rgba(s["reference"]), scale),
            composite_on_checkerboard(pose_tensor_to_pil(s["pose_map"]), scale),
            composite_on_checkerboard(tensor_to_pil_rgba(s["target"]), scale),
        ]
        for j, im in enumerate(imgs):
            x = pad + j * (cell + pad)
            canvas.paste(im, (x, y))

    os.makedirs(config.PREPARED_DIR, exist_ok=True)
    canvas.save(config.SANITY_CHECK_OUTPUT)
    print(f"\nApercu visuel sauvegarde : {config.SANITY_CHECK_OUTPUT}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run(n=args.n, seed=args.seed)


if __name__ == "__main__":
    main()
