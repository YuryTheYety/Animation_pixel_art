"""(CPU) Construit les paires alignees (reference, pose, cible) pour le
fine-tuning "style perso" : un personnage "sans features" dessine a la main
(8 directions x plusieurs animations), decline en plusieurs identites de
couleur via palette_swap.py. Meme format de sortie que data_prep.py (LPC) --
dataset.py / train.py / generate.py fonctionnent sans modification.

IMPORTANT : pour ne pas ecraser le dataset LPC deja pretraite, lancer ce
script avec PIXEL_ART_DATA_ROOT pointant vers un dossier DIFFERENT de celui
utilise pour LPC, ex:
    export PIXEL_ART_DATA_ROOT=/content/drive/MyDrive/pixel_art_retargeting/style_mannequin

Usage:
    python data_prep_style.py --source /chemin/mannequin --variants /chemin/variants
"""
import argparse
import json
import os

import numpy as np
import torch
from PIL import Image

import config
from palette_swap import PALETTES, generate_variants

DIRECTIONS = ["north", "north-east", "east", "south-east", "south",
              "south-west", "west", "north-west"]
ANIMATION_CATEGORIES = ["rotations", "Breathing_Idle", "Walking", "Running"]
REFERENCE_DIRECTION = "south"  # doit etre dans DIRECTIONS


def build_grid(source_dir):
    """Enumere (row, col) -> chemin relatif de frame, de facon deterministe.
    Une "row" = une (categorie, direction) ; une "col" = l'index de frame
    dans cette sequence (0 pour rotations, qui n'a qu'une frame statique)."""
    coords = []
    coord_to_path = {}
    row_labels = {}
    reference_coord = None
    row = 0
    for category in ANIMATION_CATEGORIES:
        for direction in DIRECTIONS:
            if category == "rotations":
                rel_path = f"rotations/{direction}.png"
                if not os.path.exists(os.path.join(source_dir, rel_path)):
                    continue
                coords.append((row, 0))
                coord_to_path[(row, 0)] = rel_path
                row_labels[row] = f"rotations/{direction}"
                if direction == REFERENCE_DIRECTION:
                    reference_coord = (row, 0)
            else:
                dir_path = os.path.join(source_dir, "animations", category, direction)
                if not os.path.isdir(dir_path):
                    continue
                frame_files = sorted(f for f in os.listdir(dir_path) if f.endswith(".png"))
                for col, fname in enumerate(frame_files):
                    rel_path = f"animations/{category}/{direction}/{fname}"
                    coords.append((row, col))
                    coord_to_path[(row, col)] = rel_path
                row_labels[row] = f"{category}/{direction}"
            row += 1

    if reference_coord is None:
        raise RuntimeError(
            f"Pose de reference introuvable : rotations/{REFERENCE_DIRECTION}.png "
            f"doit exister dans {source_dir}."
        )
    return coords, coord_to_path, row_labels, reference_coord


def load_frame_as_tensor(path):
    img = Image.open(path).convert("RGBA")
    if img.size != (config.FRAME_SIZE, config.FRAME_SIZE):
        img = img.resize((config.FRAME_SIZE, config.FRAME_SIZE), Image.NEAREST)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def rgba_to_pose_map(path):
    img = Image.open(path).convert("RGBA")
    if img.size != (config.FRAME_SIZE, config.FRAME_SIZE):
        img = img.resize((config.FRAME_SIZE, config.FRAME_SIZE), Image.NEAREST)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    r, g, b, a = arr[..., 0], arr[..., 1], arr[..., 2], arr[..., 3]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    pose = gray * a
    return torch.from_numpy(pose).unsqueeze(0).contiguous()


def prepare(source_dir, variants_dir, force=False):
    if os.path.exists(config.MANIFEST_PATH) and not force:
        print(f"Manifest deja present ({config.MANIFEST_PATH}). "
              f"Rien a faire (utiliser --force pour tout regenerer).")
        return

    if not os.path.isdir(variants_dir) or force:
        print(f"Generation des {len(PALETTES)} variantes de couleur ...")
        generate_variants(source_dir, variants_dir)

    coords, coord_to_path, row_labels, reference_coord = build_grid(source_dir)
    pose_coords = [c for c in coords if c != reference_coord]
    print(f"Grille : {len(coords)} frames ({len(row_labels)} sequences direction/animation). "
          f"Reference = {reference_coord} ({row_labels[reference_coord[0]]}).")

    print("Construction des cartes de pose (personnage source, identite retiree) ...")
    pose_maps = {}
    for coord in coords:
        pose_maps[coord] = rgba_to_pose_map(os.path.join(source_dir, coord_to_path[coord]))
    os.makedirs(config.PREPARED_DIR, exist_ok=True)
    torch.save(
        {"coords": coords, "maps": pose_maps, "mode": config.POSE_MODE},
        config.POSE_MAPS_PATH,
    )
    print(f"  -> sauvegarde : {config.POSE_MAPS_PATH}")

    os.makedirs(config.CHARACTERS_DIR, exist_ok=True)
    character_dirs = {"original": source_dir}
    for name, *_ in PALETTES:
        character_dirs[name] = os.path.join(variants_dir, name)

    manifest_characters = []
    for name, char_dir in character_dirs.items():
        reference = load_frame_as_tensor(os.path.join(char_dir, coord_to_path[reference_coord]))
        targets = {c: load_frame_as_tensor(os.path.join(char_dir, coord_to_path[c])) for c in pose_coords}
        out_path = os.path.join(config.CHARACTERS_DIR, f"{name}.pt")
        torch.save(
            {
                "folder": "mannequin",
                "variant": name,
                "reference_coord": reference_coord,
                "reference": reference,
                "targets": targets,
            },
            out_path,
        )
        manifest_characters.append({
            "id": name,
            "folder": "mannequin",
            "variant": name,
            "file": os.path.relpath(out_path, config.PREPARED_DIR),
        })
        print(f"  {name}: ok")

    manifest = {
        "frame_size": config.FRAME_SIZE,
        "reference_coord": list(reference_coord),
        "pose_coords": [list(c) for c in pose_coords],
        "pose_mode": config.POSE_MODE,
        "pose_maps_file": os.path.relpath(config.POSE_MAPS_PATH, config.PREPARED_DIR),
        "row_labels": {str(k): v for k, v in row_labels.items()},
        "characters": manifest_characters,
    }
    with open(config.MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)

    n_pairs = len(manifest_characters) * len(pose_coords)
    print(f"\nTermine. {len(manifest_characters)} personnages x {len(pose_coords)} poses "
          f"= {n_pairs} paires.")
    print(f"Manifest : {config.MANIFEST_PATH}")
    print(f"\nRappel : DATA_ROOT actuel = {config.DATA_ROOT}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Dossier du personnage source (featureless).")
    parser.add_argument("--variants", required=True, help="Dossier ou lire/ecrire les variantes de couleur.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    prepare(args.source, args.variants, force=args.force)


if __name__ == "__main__":
    main()
