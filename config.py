"""Hyperparametres et chemins partages par tout le projet.

Aucune logique ici : uniquement des constantes. Les modules CPU
(data_prep, sanity_check) et GPU (train) lisent ce fichier.
"""
import os

# ---------------------------------------------------------------------------
# Chemins
# ---------------------------------------------------------------------------
# En Colab, /content/drive existe une fois le Drive monte : on y pointe pour
# que le dataset pretraite et les checkpoints survivent aux coupures de
# session. En local (dev/tests hors Colab), on retombe sur un dossier
# ./data a cote du repo. Surchargeable via la variable d'env
# PIXEL_ART_DATA_ROOT.


def _default_data_root():
    drive_root = "/content/drive/MyDrive/pixel_art_retargeting"
    if os.path.isdir("/content/drive"):
        return drive_root
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


DATA_ROOT = os.environ.get("PIXEL_ART_DATA_ROOT", _default_data_root())

RAW_SHEETS_DIR = os.path.join(DATA_ROOT, "raw_sheets")
PREPARED_DIR = os.path.join(DATA_ROOT, "prepared")
CHARACTERS_DIR = os.path.join(PREPARED_DIR, "characters")
POSE_MAPS_PATH = os.path.join(PREPARED_DIR, "pose_maps.pt")
MANIFEST_PATH = os.path.join(PREPARED_DIR, "manifest.json")
SANITY_CHECK_OUTPUT = os.path.join(PREPARED_DIR, "sanity_check.png")

CHECKPOINT_DIR = os.path.join(DATA_ROOT, "checkpoints")
LAST_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "last.pt")
BEST_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "best.pt")

# ---------------------------------------------------------------------------
# Source de donnees : Universal LPC Spritesheet Character Generator
# ---------------------------------------------------------------------------
LPC_BASE_URL = (
    "https://raw.githubusercontent.com/sanderfrenken/"
    "Universal-LPC-Spritesheet-Character-Generator/master/"
)
LPC_BODY_PATH_TEMPLATE = "spritesheets/body/bodies/{folder}/{variant}.png"

# Corps disponibles (cf. sheet_definitions/body.json en amont). Chaque
# combinaison (folder, variant) qui existe reellement chez l'upstream
# devient un "personnage" : ils partagent tous exactement la meme grille
# de frames (c'est la propriete cle qui rend les paires alignables).
LPC_BODY_FOLDERS = ["male", "female", "muscular", "teen", "pregnant", "child"]
LPC_BODY_VARIANTS = [
    "light", "amber", "olive", "taupe", "bronze", "brown", "black",
    "lavender", "blue", "zombie_green", "green", "pale_green",
    "bright_green", "dark_green", "fur_black", "fur_brown", "fur_tan",
    "fur_copper", "fur_gold", "fur_grey", "fur_white",
]

# Personnage neutre utilise pour deriver les cartes de pose (identite
# retiree : on n'en garde que les niveaux de gris + alpha). Doit faire
# partie de LPC_BODY_FOLDERS x LPC_BODY_VARIANTS.
NEUTRAL_BASE = ("male", "light")

# Nombre max de personnages traites par data_prep.py (None = tous les
# combos disponibles). Utile pour des runs de test rapides.
MAX_CHARACTERS = None

# Nombre de sheets echantillonnes pour determiner quelles cellules de la
# grille contiennent reellement une frame (vs. cases vides/transparentes
# du template universel).
N_SAMPLE_SHEETS_FOR_VALID_MASK = 6

# ---------------------------------------------------------------------------
# Grille de frames / image
# ---------------------------------------------------------------------------
FRAME_SIZE = 64  # cote d'une frame carree, en pixels, RGBA
# Fraction minimale de pixels opaques (alpha > ALPHA_OPAQUE_THRESHOLD) dans
# une cellule 64x64 pour qu'elle soit consideree comme une frame "utilisee"
# plutot qu'une case vide du template.
ALPHA_OPAQUE_THRESHOLD = 10  # sur 255
MIN_OPAQUE_FRACTION = 0.02

# Mode de derivation de la carte de pose a partir du personnage neutre.
# "grayscale" : luminance (0..1) multipliee par l'alpha -> 1 canal.
POSE_MODE = "grayscale"
POSE_CHANNELS = 1

# ---------------------------------------------------------------------------
# Reseaux (utilise a partir du jalon 2)
# ---------------------------------------------------------------------------
GENERATOR_IN_CHANNELS = POSE_CHANNELS + 4  # [pose_map, sprite_reference(RGBA)]
GENERATOR_OUT_CHANNELS = 4  # sprite cible en RGBA
UNET_BASE_CHANNELS = 64
PATCHGAN_BASE_CHANNELS = 64

# ---------------------------------------------------------------------------
# Entrainement (utilise a partir du jalon 3)
# ---------------------------------------------------------------------------
BATCH_SIZE = 16
LEARNING_RATE_G = 2e-4
LEARNING_RATE_D = 2e-4
ADAM_BETA1 = 0.5
ADAM_BETA2 = 0.999
LAMBDA_L1 = 100.0
USE_AMP = True
TOTAL_STEPS = 100_000
OVERFIT_NUM_CHARACTERS = 20  # jalon 3 : overfit volontaire

# ~10 min de checkpoint : a ajuster selon la vitesse mesuree sur T4.
# Les deux triggers sont actifs en meme temps (le premier declenche) : le
# nombre de steps est une premiere approximation, le temps ecoule est le
# filet de securite qui garantit le "~10 min" quelle que soit la vitesse
# reelle du hardware.
CHECKPOINT_EVERY_STEPS = 500
CHECKPOINT_EVERY_SECONDS = 600

SEED = 42
