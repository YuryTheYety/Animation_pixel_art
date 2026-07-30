"""GAN conditionnel type pix2pix : generateur U-Net + discriminateur
PatchGAN. Pensé pour des sprites RGBA 64x64.

- UNetGenerator prend en entree la concatenation sur les canaux de
  [carte_de_pose, sprite_de_reference] et produit le sprite RGBA reconstruit
  dans la pose demandee.
- PatchDiscriminator est conditionnel : il juge le couple
  [condition (pose+reference), image (cible reelle ou generee)].
"""
import torch
import torch.nn as nn

import config


def build_generator_input(pose_map, reference):
    """Concatene [carte_de_pose, sprite_de_reference] sur les canaux.
    Fonction isolee (plutot qu'inline dans forward) pour pouvoir verifier
    explicitement, en test, que l'ordre des canaux est le bon."""
    return torch.cat([pose_map, reference], dim=1)


def build_discriminator_input(pose_map, reference, image):
    """Concatene la condition (pose + reference) et l'image jugee (cible
    reelle ou sortie du generateur) : PatchGAN conditionnel."""
    condition = build_generator_input(pose_map, reference)
    return torch.cat([condition, image], dim=1)


class UNetDown(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1,
                             bias=not normalize)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels, affine=True))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = self.model(x)
        return torch.cat([x, skip], dim=1)


class UNetGenerator(nn.Module):
    """U-Net 64x64 -> 1x1 -> 64x64 (6 niveaux), a la pix2pix. InstanceNorm
    plutot que BatchNorm : reste stable meme avec de petits batchs (jalon 3,
    overfit sur ~20 personnages)."""

    def __init__(self,
                 in_channels=config.GENERATOR_IN_CHANNELS,
                 out_channels=config.GENERATOR_OUT_CHANNELS,
                 base=config.UNET_BASE_CHANNELS):
        super().__init__()
        self.pose_channels = config.POSE_CHANNELS
        self.reference_channels = out_channels  # sprite reference = RGBA

        self.down1 = UNetDown(in_channels, base, normalize=False)  # 64->32
        self.down2 = UNetDown(base, base * 2)                       # 32->16
        self.down3 = UNetDown(base * 2, base * 4)                   # 16->8
        self.down4 = UNetDown(base * 4, base * 8)                   # 8->4
        self.down5 = UNetDown(base * 8, base * 8)                   # 4->2
        self.down6 = UNetDown(base * 8, base * 8, normalize=False)  # 2->1 (bottleneck)

        self.up1 = UNetUp(base * 8, base * 8, dropout=0.5)      # 1->2,  cat d5
        self.up2 = UNetUp(base * 16, base * 8, dropout=0.5)     # 2->4,  cat d4
        self.up3 = UNetUp(base * 16, base * 4)                  # 4->8,  cat d3
        self.up4 = UNetUp(base * 8, base * 2)                   # 8->16, cat d2
        self.up5 = UNetUp(base * 4, base)                       # 16->32,cat d1

        self.final = nn.Sequential(
            nn.ConvTranspose2d(base * 2, out_channels, 4, stride=2, padding=1),  # 32->64
            nn.Sigmoid(),  # sorties normalisees [0,1], comme les tenseurs prepares
        )

    def forward(self, pose_map, reference):
        x = build_generator_input(pose_map, reference)

        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)

        u1 = self.up1(d6, d5)
        u2 = self.up2(u1, d4)
        u3 = self.up3(u2, d3)
        u4 = self.up4(u3, d2)
        u5 = self.up5(u4, d1)

        return self.final(u5)


class PatchDiscriminator(nn.Module):
    """PatchGAN conditionnel (a la pix2pix) : C64-C128-C256-C512, les deux
    dernieres couches en stride 1 pour elargir le champ recepteur sans trop
    reduire la resolution spatiale (image native 64x64, deja petite)."""

    def __init__(self,
                 condition_channels=config.GENERATOR_IN_CHANNELS,
                 image_channels=config.GENERATOR_OUT_CHANNELS,
                 base=config.PATCHGAN_BASE_CHANNELS):
        super().__init__()
        in_channels = condition_channels + image_channels

        def block(in_ch, out_ch, stride, normalize=True):
            layers = [nn.Conv2d(in_ch, out_ch, 4, stride=stride, padding=1,
                                 bias=not normalize)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_ch, affine=True))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        layers += block(in_channels, base, stride=2, normalize=False)  # 64->32
        layers += block(base, base * 2, stride=2)                       # 32->16
        layers += block(base * 2, base * 4, stride=2)                   # 16->8
        layers += block(base * 4, base * 8, stride=1)                   # 8->7
        layers += [nn.Conv2d(base * 8, 1, 4, stride=1, padding=1)]      # 7->6, logits

        self.model = nn.Sequential(*layers)

    def forward(self, pose_map, reference, image):
        x = build_discriminator_input(pose_map, reference, image)
        return self.model(x)  # logits par patch (pas de sigmoid : a coupler a BCEWithLogitsLoss)


def _load_real_minibatch(batch_size):
    """Charge un vrai mini-batch depuis les donnees pretraitees du jalon 1
    si elles existent, sinon retombe sur du bruit avec les bonnes shapes."""
    import json
    import os
    import random

    if not os.path.exists(config.MANIFEST_PATH):
        return None

    with open(config.MANIFEST_PATH) as f:
        manifest = json.load(f)
    pose_maps_blob = torch.load(config.POSE_MAPS_PATH, weights_only=False)
    pose_maps = pose_maps_blob["maps"]
    pose_coords = [tuple(c) for c in manifest["pose_coords"]]

    rng = random.Random(0)
    chars = rng.sample(manifest["characters"], min(batch_size, len(manifest["characters"])))

    pose_batch, ref_batch, target_batch = [], [], []
    for char_entry in chars:
        blob = torch.load(os.path.join(config.PREPARED_DIR, char_entry["file"]), weights_only=False)
        coord = rng.choice(pose_coords)
        pose_batch.append(pose_maps[coord])
        ref_batch.append(blob["reference"])
        target_batch.append(blob["targets"][coord])

    return (torch.stack(pose_batch), torch.stack(ref_batch), torch.stack(target_batch))


def smoke_test(batch_size=4):
    """(CPU) Forward pass sur un mini-batch : verifie les shapes et que le
    conditionnement (concatenation des canaux) est correctement cable."""
    torch.manual_seed(config.SEED)

    real = _load_real_minibatch(batch_size)
    if real is not None:
        pose_map, reference, target = real
        print(f"Mini-batch reel charge depuis {config.PREPARED_DIR} "
              f"(jalon 1) : {pose_map.shape[0]} exemples.")
    else:
        pose_map = torch.rand(batch_size, config.POSE_CHANNELS, config.FRAME_SIZE, config.FRAME_SIZE)
        reference = torch.rand(batch_size, config.GENERATOR_OUT_CHANNELS, config.FRAME_SIZE, config.FRAME_SIZE)
        target = torch.rand(batch_size, config.GENERATOR_OUT_CHANNELS, config.FRAME_SIZE, config.FRAME_SIZE)
        print("Aucune donnee pretraitee trouvee (lancer data_prep.py pour le jalon 1) "
              "-> test avec des tenseurs aleatoires de shape correcte.")

    # --- 1. le conditionnement est bien l'ordre [pose, reference], pas melange ---
    cond_input = build_generator_input(pose_map, reference)
    assert cond_input.shape == (
        pose_map.shape[0], config.GENERATOR_IN_CHANNELS, config.FRAME_SIZE, config.FRAME_SIZE
    ), f"shape d'entree generateur inattendue: {tuple(cond_input.shape)}"
    assert torch.equal(cond_input[:, :config.POSE_CHANNELS], pose_map), (
        "les premiers canaux de l'entree generateur ne sont pas la carte de pose"
    )
    assert torch.equal(cond_input[:, config.POSE_CHANNELS:], reference), (
        "les canaux suivants de l'entree generateur ne sont pas le sprite de reference"
    )
    print(f"[OK] conditionnement generateur : entree = {tuple(cond_input.shape)} "
          f"= [pose({config.POSE_CHANNELS}ch), reference({config.GENERATOR_OUT_CHANNELS}ch)]")

    # --- 2. forward generateur ---
    G = UNetGenerator()
    G.eval()
    with torch.no_grad():
        generated = G(pose_map, reference)
    assert generated.shape == target.shape, (
        f"sortie generateur {tuple(generated.shape)} != shape cible {tuple(target.shape)}"
    )
    assert generated.min() >= 0.0 and generated.max() <= 1.0, (
        f"sortie generateur hors de [0,1] : min={generated.min():.3f} max={generated.max():.3f}"
    )
    n_params_g = sum(p.numel() for p in G.parameters())
    print(f"[OK] UNetGenerator forward : {tuple(pose_map.shape)} + {tuple(reference.shape)} "
          f"-> {tuple(generated.shape)}  ({n_params_g:,} parametres)")

    # --- 3. forward discriminateur, sur la vraie cible et sur la sortie generee ---
    D = PatchDiscriminator()
    D.eval()
    with torch.no_grad():
        patch_real = D(pose_map, reference, target)
        patch_fake = D(pose_map, reference, generated)
    assert patch_real.shape == patch_fake.shape, (
        "le discriminateur doit produire la meme shape sur reel et genere"
    )
    n_params_d = sum(p.numel() for p in D.parameters())
    print(f"[OK] PatchDiscriminator forward : condition+image -> patch logits "
          f"{tuple(patch_real.shape)}  ({n_params_d:,} parametres)")

    print("\nJalon 2 : forward pass CPU + conditionnement OK.")


if __name__ == "__main__":
    smoke_test()
