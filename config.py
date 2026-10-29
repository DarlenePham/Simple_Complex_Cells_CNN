
from pathlib import Path

# ==========================================
# Project paths
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parent

DATASET_ROOT = PROJECT_ROOT / "BSDS300"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINT_PATH = CHECKPOINT_DIR / "simple_complex_best.pth"

# ==========================================
# Dataset configuration
# ==========================================
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 384

ANNOTATION_TYPE = "gray"
VALIDATION_FRACTION = 0.10

# ==========================================
# Training configuration
# ==========================================
EPOCHS = 30
BATCH_SIZE = 2
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
POSITIVE_WEIGHT = 5.0

# ==========================================
# Gabor / simple-complex cell configuration
# ==========================================
GABOR_SIGMA = 2.5
GABOR_WAVELENGTH = 5.0
GABOR_GAMMA = 0.5

# Eight orientations, evenly spaced from 0 to 157.5 degrees
NUM_ORIENTATIONS = 8
ORIENTATIONS = tuple(
    i * 180.0 / NUM_ORIENTATIONS
    for i in range(NUM_ORIENTATIONS)
)

# ==========================================
# CNN configuration
# ==========================================
HIDDEN_CHANNELS = 32

# ==========================================
# Reproducibility and device
# ==========================================
RANDOM_SEED = 42
NUM_WORKERS = 0  # Safe default for Windows

# ==========================================
# Learning-rate scheduler
# ==========================================
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 3
MIN_LEARNING_RATE = 1e-6
