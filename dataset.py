"""Dataset PyTorch qui charge les paires (pose_map, reference, cible)
pretraitees par data_prep.py (jalon 1)."""
import json
import os

import torch
from torch.utils.data import Dataset

import config


class PairDataset(Dataset):
    """Un exemple = (pose_map, reference, target) pour un personnage et une
    pose cible donnes. `character_ids` / `pose_coords` permettent de
    restreindre le dataset (ex: overfit du jalon 3 sur ~20 personnages)."""

    def __init__(self, character_ids=None, pose_coords=None):
        if not os.path.exists(config.MANIFEST_PATH):
            raise FileNotFoundError(
                f"{config.MANIFEST_PATH} introuvable : lancer data_prep.py d'abord."
            )
        with open(config.MANIFEST_PATH) as f:
            manifest = json.load(f)

        self.characters = manifest["characters"]
        if character_ids is not None:
            wanted = set(character_ids)
            self.characters = [c for c in self.characters if c["id"] in wanted]
            if not self.characters:
                raise ValueError("Aucun des character_ids demandes n'est dans le manifest.")

        all_pose_coords = [tuple(c) for c in manifest["pose_coords"]]
        self.pose_coords = pose_coords if pose_coords is not None else all_pose_coords
        self.reference_coord = tuple(manifest["reference_coord"])
        self.frame_size = manifest["frame_size"]

        pose_maps_blob = torch.load(config.POSE_MAPS_PATH, weights_only=False)
        self.pose_maps = pose_maps_blob["maps"]

        self._char_cache = {}
        self.index = [
            (ci, coord)
            for ci in range(len(self.characters))
            for coord in self.pose_coords
        ]

    def character_ids(self):
        return [c["id"] for c in self.characters]

    def _load_character(self, ci):
        if ci not in self._char_cache:
            entry = self.characters[ci]
            path = os.path.join(config.PREPARED_DIR, entry["file"])
            self._char_cache[ci] = torch.load(path, weights_only=False)
        return self._char_cache[ci]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        ci, coord = self.index[idx]
        blob = self._load_character(ci)
        return {
            "pose_map": self.pose_maps[coord],
            "reference": blob["reference"],
            "target": blob["targets"][coord],
            "character_id": blob["folder"] + "_" + blob["variant"],
            "pose_coord": coord,
        }
