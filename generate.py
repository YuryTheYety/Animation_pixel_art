"""Inference : genere une sequence d'images (une animation) a partir d'un
generateur entraine, en enchainant les frames d'une meme ligne de la grille
LPC (= une animation/direction) sur une reference fixe. Aucun entrainement :
pur forward pass repete, comme prevu par le brief ("une fois le single-frame
propre, on enchaine les frames").

Usage:
    # lister les animations (lignes) disponibles et leur nombre de frames
    python generate.py --list-rows

    # animer un personnage du dataset prepare
    python generate.py --character-id male_light --row 0 --out anim.gif

    # animer une image externe (sprite hors dataset, cf. limites : le
    # generateur reste calibre sur la distribution LPC / celle du
    # fine-tuning eventuel)
    python generate.py --reference-image mon_sprite.png --row 0 --out anim.gif

    # composer une couche (cheveux, vetements...) par-dessus le corps genere.
    # Le dossier d'overlay doit suivre la meme structure que le personnage
    # source du fine-tuning (rotations/<dir>.png + animations/<anim>/<dir>/
    # frame_NNN.png), dessine pose par pose pour suivre le mouvement du corps
    # -- aucun apprentissage necessaire, pur compositing alpha.
    python generate.py --character-id rouge_brique --row 20 --overlay mes_cheveux/ --out anim.gif
"""
import argparse
import json
import os

import numpy as np
import torch
from PIL import Image

import config
from models import UNetGenerator
from sanity_check import composite_on_checkerboard, tensor_to_pil_rgba
from train import get_device
from data_prep_style import build_grid


def load_manifest():
    if not os.path.exists(config.MANIFEST_PATH):
        raise FileNotFoundError(
            f"{config.MANIFEST_PATH} introuvable : lancer data_prep.py d'abord."
        )
    with open(config.MANIFEST_PATH) as f:
        return json.load(f)


def load_generator(checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"{checkpoint_path} introuvable : lancer train.py d'abord (jalon 3/4)."
        )
    G = UNetGenerator().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    G.load_state_dict(ckpt["generator"])
    G.eval()
    return G, ckpt.get("step")


def load_reference_from_character(character_id, manifest):
    entry = next((c for c in manifest["characters"] if c["id"] == character_id), None)
    if entry is None:
        available = ", ".join(c["id"] for c in manifest["characters"][:10])
        raise ValueError(
            f"'{character_id}' absent du manifest. Exemples disponibles: {available}, ..."
        )
    blob = torch.load(os.path.join(config.PREPARED_DIR, entry["file"]), weights_only=False)
    return blob["reference"]


def load_reference_from_image(path):
    """Charge une image externe comme sprite de reference. Resize en
    nearest-neighbor uniquement (jamais d'interpolation qui lisserait le
    pixel art) -- mais voir les limites documentees dans le README/plan :
    le generateur reste calibre sur la distribution LPC (ou celle du
    fine-tuning), un sprite hors distribution peut donner un resultat
    degrade meme avec un cadrage correct."""
    img = Image.open(path).convert("RGBA")
    if img.size != (config.FRAME_SIZE, config.FRAME_SIZE):
        print(f"[!] {path} fait {img.size}, redimensionne en "
              f"{config.FRAME_SIZE}x{config.FRAME_SIZE} (nearest-neighbor). "
              f"Verifie que le personnage est bien cadre/pose comme une "
              f"reference LPC (cf. sanity_check.py pour comparer).")
        img = img.resize((config.FRAME_SIZE, config.FRAME_SIZE), Image.NEAREST)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def list_rows(manifest):
    all_coords = [tuple(manifest["reference_coord"])] + [tuple(c) for c in manifest["pose_coords"]]
    by_row = {}
    for r, c in all_coords:
        by_row.setdefault(r, []).append(c)
    row_labels = manifest.get("row_labels", {})
    print(f"{len(by_row)} lignes (animations/directions) disponibles :")
    for row in sorted(by_row):
        cols = sorted(by_row[row])
        label = row_labels.get(str(row))
        label_str = f"  [{label}]" if label else ""
        print(f"  row={row:<3} {len(cols)} frames  cols={cols}{label_str}")


def animation_coords(row, manifest):
    all_coords = [tuple(manifest["reference_coord"])] + [tuple(c) for c in manifest["pose_coords"]]
    coords = sorted(c for r, c in all_coords if r == row)
    if not coords:
        raise ValueError(f"Aucune frame valide sur la ligne {row}. Essaie --list-rows.")
    return [(row, c) for c in coords]


def load_overlay_grid(overlay_dir):
    """Reutilise la meme logique de grille que data_prep_style.py : si le
    dossier d'overlay a la meme structure (rotations/ + animations/), les
    coordonnees (row,col) tombent naturellement sur les memes poses."""
    _, coord_to_path, _, _ = build_grid(overlay_dir)
    return coord_to_path


def composite_overlay(body_tensor, overlay_dir, coord_to_path, coord):
    """Colle la frame d'overlay correspondant a cette pose par-dessus le
    corps genere (alpha compositing simple, aucun apprentissage). Renvoie
    le corps tel quel si l'overlay n'a pas de frame pour cette pose."""
    if coord not in coord_to_path:
        return body_tensor, False
    overlay_path = os.path.join(overlay_dir, coord_to_path[coord])
    overlay_img = Image.open(overlay_path).convert("RGBA")
    if overlay_img.size != (config.FRAME_SIZE, config.FRAME_SIZE):
        overlay_img = overlay_img.resize((config.FRAME_SIZE, config.FRAME_SIZE), Image.NEAREST)
    body_img = tensor_to_pil_rgba(body_tensor)
    body_img.alpha_composite(overlay_img)
    arr = np.asarray(body_img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous(), True


def apply_overlay(frames, coords, overlay_dir):
    coord_to_path = load_overlay_grid(overlay_dir)
    composited, n_hit = [], 0
    for frame, coord in zip(frames, coords):
        new_frame, hit = composite_overlay(frame, overlay_dir, coord_to_path, coord)
        composited.append(new_frame)
        n_hit += hit
    if n_hit == 0:
        print(f"[!] Aucune frame d'overlay trouvee pour ces poses dans {overlay_dir} "
              f"-- verifier que sa structure de dossiers correspond a celle du "
              f"personnage source (rotations/ + animations/<anim>/<direction>/).")
    elif n_hit < len(frames):
        print(f"[!] Overlay applique sur {n_hit}/{len(frames)} frames seulement "
              f"(poses manquantes dans {overlay_dir}).")
    else:
        print(f"Overlay applique sur les {n_hit} frames.")
    return composited


def generate_sequence(G, pose_maps, reference, coords, device):
    frames = []
    with torch.no_grad():
        ref_batch = reference.unsqueeze(0).to(device)
        for coord in coords:
            pose_batch = pose_maps[coord].unsqueeze(0).to(device)
            out = G(pose_batch, ref_batch)[0].detach().cpu().clamp(0, 1)
            frames.append(out)
    return frames


def save_spritesheet(frames, path):
    """Planche horizontale, resolution native 64x64/frame, alpha preservee
    -- directement reutilisable dans un moteur de jeu."""
    w, h = config.FRAME_SIZE, config.FRAME_SIZE
    sheet = Image.new("RGBA", (w * len(frames), h), (0, 0, 0, 0))
    for i, frame in enumerate(frames):
        sheet.paste(tensor_to_pil_rgba(frame), (i * w, 0))
    sheet.save(path)


def save_gif_preview(frames, path, scale=6, duration_ms=120):
    """Apercu visuel (fond damier, upscale nearest-neighbor) -- pas destine
    a etre reintegre dans un moteur de jeu, juste a etre regarde."""
    composited = [composite_on_checkerboard(tensor_to_pil_rgba(f), scale) for f in frames]
    composited[0].save(
        path, save_all=True, append_images=composited[1:],
        duration=duration_ms, loop=0,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", choices=["best", "last"], default="best")
    parser.add_argument("--character-id", type=str, default=None,
                         help="Personnage du dataset prepare a utiliser comme reference.")
    parser.add_argument("--reference-image", type=str, default=None,
                         help="Image externe (RGBA de preference) a utiliser comme reference.")
    parser.add_argument("--row", type=int, default=None,
                         help="Ligne de la grille = une animation/direction. Voir --list-rows.")
    parser.add_argument("--list-rows", action="store_true",
                         help="Liste les lignes/animations disponibles et sort.")
    parser.add_argument("--out", type=str, default=None,
                         help="Chemin de sortie .gif (apercu) ou .png (planche native).")
    parser.add_argument("--overlay", type=str, default=None,
                         help="Dossier d'une couche (cheveux, vetements...) a composer "
                              "par-dessus le corps genere, meme structure que le "
                              "personnage source (rotations/ + animations/).")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    manifest = load_manifest()

    if args.list_rows:
        list_rows(manifest)
        return

    if args.row is None:
        raise SystemExit("--row est requis (utiliser --list-rows pour voir les options).")
    if bool(args.character_id) == bool(args.reference_image):
        raise SystemExit("Fournir exactement un de --character-id ou --reference-image.")

    device = get_device(args.device)
    checkpoint_path = config.BEST_CHECKPOINT_PATH if args.checkpoint == "best" else config.LAST_CHECKPOINT_PATH
    G, step = load_generator(checkpoint_path, device)
    print(f"Generateur charge depuis {checkpoint_path} (step={step}), device={device}.")

    pose_maps_blob = torch.load(config.POSE_MAPS_PATH, weights_only=False)
    pose_maps = pose_maps_blob["maps"]

    if args.character_id:
        reference = load_reference_from_character(args.character_id, manifest)
    else:
        reference = load_reference_from_image(args.reference_image)

    coords = animation_coords(args.row, manifest)
    print(f"Animation ligne {args.row} : {len(coords)} frames.")
    frames = generate_sequence(G, pose_maps, reference, coords, device)

    if args.overlay:
        frames = apply_overlay(frames, coords, args.overlay)

    out = args.out or os.path.join(config.PREPARED_DIR, f"generated_row{args.row}.gif")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    if out.lower().endswith(".png"):
        save_spritesheet(frames, out)
    else:
        save_gif_preview(frames, out)
    print(f"Sortie : {out}")


if __name__ == "__main__":
    main()
