"""Boucle d'entrainement pix2pix (GPU en Colab). AMP, checkpoint/resume
Drive, logging. Resume-first : au demarrage, charge automatiquement le
dernier checkpoint et reprend la ou l'entrainement s'etait arrete.

Jalon 3 (overfit volontaire) :
    python train.py --num-characters 20 --steps 3000

Jalon 4 (scaling), une fois l'overfit valide : relancer sans
--num-characters / --num-poses pour utiliser tout le dataset pretraite.
"""
import argparse
import json
import os
import tempfile
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from dataset import PairDataset
from models import PatchDiscriminator, UNetGenerator
from sanity_check import composite_on_checkerboard, pose_tensor_to_pil, tensor_to_pil_rgba
from PIL import Image


def get_device(requested=None):
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def atomic_save(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(path))
    os.close(fd)
    try:
        torch.save(obj, tmp_path)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def build_checkpoint(step, best_score, G, D, optG, optD, scalerG, scalerD, use_amp):
    ckpt = {
        "step": step,
        "best_score": best_score,
        "generator": G.state_dict(),
        "discriminator": D.state_dict(),
        "optimizer_g": optG.state_dict(),
        "optimizer_d": optD.state_dict(),
        "use_amp": use_amp,
    }
    if use_amp:
        ckpt["scaler_g"] = scalerG.state_dict()
        ckpt["scaler_d"] = scalerD.state_dict()
    return ckpt


def load_checkpoint(path, G, D, optG, optD, scalerG, scalerD, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    G.load_state_dict(ckpt["generator"])
    D.load_state_dict(ckpt["discriminator"])
    optG.load_state_dict(ckpt["optimizer_g"])
    optD.load_state_dict(ckpt["optimizer_d"])
    if ckpt.get("use_amp") and "scaler_g" in ckpt:
        scalerG.load_state_dict(ckpt["scaler_g"])
        scalerD.load_state_dict(ckpt["scaler_d"])
    return ckpt["step"], ckpt["best_score"]


def save_comparison_grid(G, dataset, device, path, n=8, seed=0):
    """(reference | pose | cible | generee) pour n exemples, comme
    sanity_check.py mais avec la sortie du generateur en plus."""
    import random
    rng = random.Random(seed)
    n = min(n, len(dataset))
    indices = rng.sample(range(len(dataset)), n)

    scale, pad, header_h = 4, 6, 20
    cell = dataset.frame_size * scale
    labels = ["reference", "pose", "cible", "generee"]
    cols = len(labels)
    W = cols * cell + (cols + 1) * pad
    H = header_h + n * cell + (n + 1) * pad
    canvas = Image.new("RGB", (W, H), (30, 30, 30))
    try:
        from PIL import ImageDraw
        draw = ImageDraw.Draw(canvas)
        for j, label in enumerate(labels):
            draw.text((pad + j * (cell + pad), 2), label, fill=(255, 255, 255))
    except Exception:
        pass

    G.eval()
    with torch.no_grad():
        for i, idx in enumerate(indices):
            sample = dataset[idx]
            pose_map = sample["pose_map"].unsqueeze(0).to(device)
            reference = sample["reference"].unsqueeze(0).to(device)
            target = sample["target"]
            generated = G(pose_map, reference)[0].cpu().clamp(0, 1)

            imgs = [
                composite_on_checkerboard(tensor_to_pil_rgba(sample["reference"]), scale),
                composite_on_checkerboard(pose_tensor_to_pil(sample["pose_map"]), scale),
                composite_on_checkerboard(tensor_to_pil_rgba(target), scale),
                composite_on_checkerboard(tensor_to_pil_rgba(generated), scale),
            ]
            y = header_h + pad + i * (cell + pad)
            for j, im in enumerate(imgs):
                canvas.paste(im, (pad + j * (cell + pad), y))
    G.train()

    os.makedirs(os.path.dirname(path), exist_ok=True)
    canvas.save(path)


def train(args):
    device = get_device(args.device)
    use_amp = config.USE_AMP and device.type == "cuda"
    torch.manual_seed(config.SEED)

    with open(config.MANIFEST_PATH) as f:
        manifest = json.load(f)
    all_char_ids = sorted(c["id"] for c in manifest["characters"])
    if args.num_characters is not None:
        char_ids = all_char_ids[: args.num_characters]
    else:
        char_ids = all_char_ids

    all_pose_coords = [tuple(c) for c in manifest["pose_coords"]]
    pose_coords = all_pose_coords[: args.num_poses] if args.num_poses else None

    dataset = PairDataset(character_ids=char_ids, pose_coords=pose_coords)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True,
    )
    print(f"Dataset: {len(char_ids)} personnages x "
          f"{len(pose_coords) if pose_coords else len(all_pose_coords)} poses "
          f"= {len(dataset)} paires. Device={device}, AMP={'on' if use_amp else 'off'}.")

    G = UNetGenerator().to(device)
    D = PatchDiscriminator().to(device)
    optG = torch.optim.Adam(G.parameters(), lr=args.lr_g, betas=(config.ADAM_BETA1, config.ADAM_BETA2))
    optD = torch.optim.Adam(D.parameters(), lr=args.lr_d, betas=(config.ADAM_BETA1, config.ADAM_BETA2))
    scalerG = torch.amp.GradScaler(device.type, enabled=use_amp)
    scalerD = torch.amp.GradScaler(device.type, enabled=use_amp)

    bce = nn.BCEWithLogitsLoss()
    l1 = nn.L1Loss()

    step, best_score = 0, float("inf")
    if os.path.exists(config.LAST_CHECKPOINT_PATH) and not args.no_resume:
        step, best_score = load_checkpoint(
            config.LAST_CHECKPOINT_PATH, G, D, optG, optD, scalerG, scalerD, device
        )
        print(f"Reprise depuis {config.LAST_CHECKPOINT_PATH} : step={step}, "
              f"best_score={best_score:.4f}")
    else:
        print("Pas de checkpoint existant (ou --no-resume) : depart a zero.")

    total_steps = args.steps
    last_checkpoint_time = time.time()
    running_l1, running_g_adv, running_d = 0.0, 0.0, 0.0
    log_every = args.log_every

    G.train()
    D.train()
    data_iter = iter(loader)
    t0 = time.time()
    first_step = step  # premiere step de CE run (utile apres une reprise)

    print("Chargement du premier batch... (peut prendre du temps la premiere "
          "fois : lecture depuis Drive de plusieurs personnages)")

    while step < total_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        pose_map = batch["pose_map"].to(device, non_blocking=True)
        reference = batch["reference"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)

        # --- Discriminateur ---
        optD.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            fake = G(pose_map, reference).detach()
            pred_real = D(pose_map, reference, target)
            pred_fake = D(pose_map, reference, fake)
            loss_d_real = bce(pred_real, torch.ones_like(pred_real))
            loss_d_fake = bce(pred_fake, torch.zeros_like(pred_fake))
            loss_d = 0.5 * (loss_d_real + loss_d_fake)
        scalerD.scale(loss_d).backward()
        scalerD.step(optD)
        scalerD.update()

        # --- Generateur ---
        optG.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            fake = G(pose_map, reference)
            pred_fake_g = D(pose_map, reference, fake)
            loss_g_adv = bce(pred_fake_g, torch.ones_like(pred_fake_g))
            loss_g_l1 = l1(fake, target)
            loss_g = loss_g_adv + config.LAMBDA_L1 * loss_g_l1
        scalerG.scale(loss_g).backward()
        scalerG.step(optG)
        scalerG.update()

        step += 1
        running_l1 += loss_g_l1.item()
        running_g_adv += loss_g_adv.item()
        running_d += loss_d.item()

        if step % log_every == 0 or step == first_step + 1:
            n = step - first_step if step == first_step + 1 else log_every
            elapsed = time.time() - t0
            print(f"step {step}/{total_steps}  L1={running_l1/n:.4f}  "
                  f"G_adv={running_g_adv/n:.4f}  D={running_d/n:.4f}  "
                  f"({elapsed/n:.3f}s/step)")
            running_l1 = running_g_adv = running_d = 0.0
            t0 = time.time()

        due_by_steps = step % args.checkpoint_every_steps == 0
        due_by_time = (time.time() - last_checkpoint_time) >= args.checkpoint_every_seconds
        if due_by_steps or due_by_time or step == total_steps:
            score = loss_g_l1.item()
            ckpt = build_checkpoint(step, best_score, G, D, optG, optD, scalerG, scalerD, use_amp)
            atomic_save(ckpt, config.LAST_CHECKPOINT_PATH)
            if score < best_score:
                best_score = score
                ckpt["best_score"] = best_score
                atomic_save(ckpt, config.BEST_CHECKPOINT_PATH)
                print(f"  [checkpoint] step={step}  meilleur score L1={best_score:.4f} "
                      f"-> {config.BEST_CHECKPOINT_PATH}")
            else:
                print(f"  [checkpoint] step={step} -> {config.LAST_CHECKPOINT_PATH}")
            last_checkpoint_time = time.time()

    preview_path = os.path.join(config.PREPARED_DIR, "train_preview.png")
    save_comparison_grid(G, dataset, device, preview_path, n=min(8, len(char_ids)))
    print(f"\nApercu genere vs cible : {preview_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=config.TOTAL_STEPS)
    parser.add_argument("--num-characters", type=int, default=None,
                         help="Restreint le dataset aux N premiers personnages "
                              "(jalon 3: 20, cf. config.OVERFIT_NUM_CHARACTERS).")
    parser.add_argument("--num-poses", type=int, default=None,
                         help="Restreint le dataset aux N premieres poses par personnage.")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr-g", type=float, default=config.LEARNING_RATE_G)
    parser.add_argument("--lr-d", type=float, default=config.LEARNING_RATE_D)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--checkpoint-every-steps", type=int, default=config.CHECKPOINT_EVERY_STEPS)
    parser.add_argument("--checkpoint-every-seconds", type=int, default=config.CHECKPOINT_EVERY_SECONDS)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no-resume", action="store_true",
                         help="Ignore le checkpoint existant et repart a zero.")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
