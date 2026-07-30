"""(CPU) Construit les paires alignees (reference, pose, cible) depuis les
spritesheets LPC et les sauvegarde pretraitees sur Drive.

A executer une seule fois (ou apres --force) : le train.py ne doit jamais
regenerer le dataset a chaque demarrage de session Colab.

Usage:
    python data_prep.py [--limit N] [--workers 8] [--force]
"""
import argparse
import io
import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
from PIL import Image

import config

USER_AGENT = "pixel-art-retargeting-data-prep/1.0"


def _download(url, timeout=15, retries=3):
    """Telecharge une URL en bytes. Renvoie None si 404 (frame/variant
    absente chez l'upstream), retente sur les autres erreurs reseau."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if attempt == retries - 1:
                raise
        except (urllib.error.URLError, TimeoutError):
            if attempt == retries - 1:
                raise
        time.sleep(2 ** attempt)
    return None


def _sheet_url(folder, variant):
    path = config.LPC_BODY_PATH_TEMPLATE.format(folder=folder, variant=variant)
    return config.LPC_BASE_URL + path


def _raw_sheet_path(folder, variant):
    return os.path.join(config.RAW_SHEETS_DIR, f"{folder}_{variant}.png")


def fetch_raw_sheet(folder, variant, force=False):
    """Telecharge (avec cache disque) une spritesheet de corps. Renvoie le
    chemin local, ou None si ce combo n'existe pas chez l'upstream."""
    dest = _raw_sheet_path(folder, variant)
    if os.path.exists(dest) and not force:
        return dest
    data = _download(_sheet_url(folder, variant))
    if data is None:
        return None
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "wb") as f:
        f.write(data)
    return dest


def download_all_sheets(combos, workers, force):
    os.makedirs(config.RAW_SHEETS_DIR, exist_ok=True)
    available, missing = [], []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(fetch_raw_sheet, folder, variant, force): (folder, variant)
            for folder, variant in combos
        }
        for fut in as_completed(futures):
            folder, variant = futures[fut]
            path = fut.result()
            if path is None:
                missing.append((folder, variant))
            else:
                available.append((folder, variant, path))
    return available, missing


def load_rgba(path):
    img = Image.open(path).convert("RGBA")
    return img


def grid_shape(img):
    w, h = img.size
    assert w % config.FRAME_SIZE == 0 and h % config.FRAME_SIZE == 0, (
        f"{img} n'est pas un multiple de {config.FRAME_SIZE}px : {w}x{h}"
    )
    return h // config.FRAME_SIZE, w // config.FRAME_SIZE  # rows, cols


def crop_frame(img, row, col):
    s = config.FRAME_SIZE
    box = (col * s, row * s, (col + 1) * s, (row + 1) * s)
    return img.crop(box)


def frame_to_rgba_tensor(frame_img):
    arr = np.asarray(frame_img, dtype=np.float32) / 255.0  # H,W,4
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # 4,H,W


def alpha_opaque_fraction(frame_img):
    alpha = np.asarray(frame_img, dtype=np.uint8)[:, :, 3]
    return float((alpha > config.ALPHA_OPAQUE_THRESHOLD).mean())


def compute_valid_frame_mask(sample_paths, rows, cols):
    """Une cellule de la grille est valide si elle contient reellement une
    frame (pas une case vide du template universel) dans TOUS les sheets
    echantillonnes (intersection -> evite les faux positifs isoles)."""
    mask = np.ones((rows, cols), dtype=bool)
    for path in sample_paths:
        img = load_rgba(path)
        r, c = grid_shape(img)
        assert (r, c) == (rows, cols), (
            f"Grille incoherente pour {path}: {(r, c)} != {(rows, cols)}"
        )
        for row in range(rows):
            for col in range(cols):
                frac = alpha_opaque_fraction(crop_frame(img, row, col))
                if frac < config.MIN_OPAQUE_FRACTION:
                    mask[row, col] = False
    return mask


def rgba_to_pose_map(frame_img):
    """Derive une carte de pose identite-neutre a partir d'une frame RGBA :
    luminance en niveaux de gris, ponderee par l'alpha (fond -> 0)."""
    arr = np.asarray(frame_img, dtype=np.float32) / 255.0  # H,W,4
    r, g, b, a = arr[..., 0], arr[..., 1], arr[..., 2], arr[..., 3]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    pose = gray * a
    return torch.from_numpy(pose).unsqueeze(0).contiguous()  # 1,H,W


def build_pose_maps(neutral_sheet_path, valid_coords):
    img = load_rgba(neutral_sheet_path)
    pose_maps = {}
    for (row, col) in valid_coords:
        pose_maps[(row, col)] = rgba_to_pose_map(crop_frame(img, row, col))
    return pose_maps


def prepare(limit=None, workers=8, force=False):
    if os.path.exists(config.MANIFEST_PATH) and not force:
        print(f"Manifest deja present ({config.MANIFEST_PATH}). "
              f"Rien a faire (utiliser --force pour tout regenerer).")
        return

    combos = [
        (folder, variant)
        for folder in config.LPC_BODY_FOLDERS
        for variant in config.LPC_BODY_VARIANTS
    ]
    if config.NEUTRAL_BASE not in combos:
        combos.insert(0, config.NEUTRAL_BASE)
    if limit is not None:
        # Toujours garder le personnage neutre dans le sous-ensemble.
        rest = [c for c in combos if c != config.NEUTRAL_BASE]
        combos = [config.NEUTRAL_BASE] + rest[: max(0, limit - 1)]

    print(f"Telechargement de {len(combos)} spritesheets (cache: "
          f"{config.RAW_SHEETS_DIR}) ...")
    available, missing = download_all_sheets(combos, workers, force)
    print(f"  -> {len(available)} disponibles, {len(missing)} absentes "
          f"chez l'upstream (ignorees).")
    if not available:
        raise RuntimeError("Aucune spritesheet telechargee, verifier le reseau.")

    neutral_entry = next(
        (e for e in available if (e[0], e[1]) == config.NEUTRAL_BASE), None
    )
    if neutral_entry is None:
        raise RuntimeError(
            f"Le personnage neutre {config.NEUTRAL_BASE} (config.NEUTRAL_BASE) "
            f"n'a pas pu etre telecharge : impossible de deriver les poses."
        )
    neutral_path = neutral_entry[2]

    first_img = load_rgba(available[0][2])
    rows, cols = grid_shape(first_img)
    print(f"Grille de frames : {rows} lignes x {cols} colonnes "
          f"({config.FRAME_SIZE}px/frame).")

    sample_paths = [p for _, _, p in available[: config.N_SAMPLE_SHEETS_FOR_VALID_MASK]]
    if neutral_path not in sample_paths:
        sample_paths.append(neutral_path)
    print(f"Calcul du masque de frames valides sur {len(sample_paths)} sheets ...")
    mask = compute_valid_frame_mask(sample_paths, rows, cols)
    valid_coords = [(r, c) for r in range(rows) for c in range(cols) if mask[r, c]]
    if not valid_coords:
        raise RuntimeError("Aucune frame valide detectee, verifier MIN_OPAQUE_FRACTION.")

    reference_coord = valid_coords[0]
    pose_coords = valid_coords[1:]
    print(f"  -> {len(valid_coords)} frames valides. Reference = {reference_coord}, "
          f"{len(pose_coords)} poses cibles possibles.")

    print("Construction des cartes de pose (personnage neutre, identite retiree) ...")
    pose_maps = build_pose_maps(neutral_path, valid_coords)
    os.makedirs(config.PREPARED_DIR, exist_ok=True)
    torch.save(
        {"coords": valid_coords, "maps": pose_maps, "mode": config.POSE_MODE},
        config.POSE_MAPS_PATH,
    )
    print(f"  -> sauvegarde : {config.POSE_MAPS_PATH}")

    os.makedirs(config.CHARACTERS_DIR, exist_ok=True)
    manifest_characters = []
    for folder, variant, path in available:
        img = load_rgba(path)
        r, c = grid_shape(img)
        if (r, c) != (rows, cols):
            print(f"  [!] grille incoherente pour {folder}/{variant}, ignore.")
            continue
        reference = frame_to_rgba_tensor(crop_frame(img, *reference_coord))
        targets = {coord: frame_to_rgba_tensor(crop_frame(img, *coord)) for coord in pose_coords}
        char_id = f"{folder}_{variant}"
        out_path = os.path.join(config.CHARACTERS_DIR, f"{char_id}.pt")
        torch.save(
            {
                "folder": folder,
                "variant": variant,
                "reference_coord": reference_coord,
                "reference": reference,
                "targets": targets,
            },
            out_path,
        )
        manifest_characters.append({
            "id": char_id,
            "folder": folder,
            "variant": variant,
            "file": os.path.relpath(out_path, config.PREPARED_DIR),
        })

    manifest = {
        "frame_size": config.FRAME_SIZE,
        "grid_rows": rows,
        "grid_cols": cols,
        "reference_coord": list(reference_coord),
        "pose_coords": [list(c) for c in pose_coords],
        "pose_mode": config.POSE_MODE,
        "pose_maps_file": os.path.relpath(config.POSE_MAPS_PATH, config.PREPARED_DIR),
        "neutral_base": list(config.NEUTRAL_BASE),
        "characters": manifest_characters,
        "missing_combos": [f"{f}/{v}" for f, v in missing],
    }
    with open(config.MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)

    n_pairs = len(manifest_characters) * len(pose_coords)
    print(f"Termine. {len(manifest_characters)} personnages x {len(pose_coords)} poses "
          f"= {n_pairs} paires (reference, pose, cible).")
    print(f"Manifest : {config.MANIFEST_PATH}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=config.MAX_CHARACTERS,
                         help="Nombre max de personnages a traiter (defaut: tous).")
    parser.add_argument("--workers", type=int, default=8,
                         help="Threads de telechargement en parallele.")
    parser.add_argument("--force", action="store_true",
                         help="Retelecharge/regenere meme si le manifest existe deja.")
    args = parser.parse_args()
    prepare(limit=args.limit, workers=args.workers, force=args.force)


if __name__ == "__main__":
    main()
