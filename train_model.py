
import random
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, random_split

import config as cfg

from Models.bsds_dataset import BSDSBoundaryDataset
from Models.complex_cell_model import SimpleComplexEdgeNet


# ============================================================
# Reproducibility and device
# ============================================================

random.seed(cfg.RANDOM_SEED)
np.random.seed(cfg.RANDOM_SEED)
torch.manual_seed(cfg.RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(cfg.RANDOM_SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ============================================================
# Dataset
# ============================================================

dataset = BSDSBoundaryDataset(
    root=cfg.DATASET_ROOT,
    split="train",
    size=(cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH),
    annotation_type=cfg.ANNOTATION_TYPE,
)

if len(dataset) < 2:
    raise ValueError(
        f"Dataset contains only {len(dataset)} samples. "
        "Check DATASET_ROOT and the dataset split."
    )

n_val = max(
    1,
    int(len(dataset) * cfg.VALIDATION_FRACTION)
)
n_train = len(dataset) - n_val

if n_train < 1:
    raise ValueError(
        "Not enough samples for training. "
        "Reduce VALIDATION_FRACTION."
    )

train_set, val_set = random_split(
    dataset,
    [n_train, n_val],
    generator=torch.Generator().manual_seed(
        cfg.RANDOM_SEED
    ),
)

train_loader = DataLoader(
    train_set,
    batch_size=cfg.BATCH_SIZE,
    shuffle=True,
    num_workers=cfg.NUM_WORKERS,
    pin_memory=(device.type == "cuda"),
)

val_loader = DataLoader(
    val_set,
    batch_size=cfg.BATCH_SIZE,
    shuffle=False,
    num_workers=cfg.NUM_WORKERS,
    pin_memory=(device.type == "cuda"),
)

print(f"Dataset size: {len(dataset)}")
print(f"Training samples: {n_train}")
print(f"Validation samples: {n_val}")


# ============================================================
# Model
# ============================================================

model = SimpleComplexEdgeNet(
    sigma=cfg.GABOR_SIGMA,
    wavelength=cfg.GABOR_WAVELENGTH,
    gamma=cfg.GABOR_GAMMA,
    orientations=cfg.ORIENTATIONS,
    hidden_channels=cfg.HIDDEN_CHANNELS,
).to(device)

print(
    f"Gabor orientations: {cfg.ORIENTATIONS}"
)
print(
    f"Gabor sigma: {cfg.GABOR_SIGMA}, "
    f"wavelength: {cfg.GABOR_WAVELENGTH}, "
    f"gamma: {cfg.GABOR_GAMMA}"
)
print(f"CNN hidden channels: {cfg.HIDDEN_CHANNELS}")


# ============================================================
# Loss function
# ============================================================

criterion = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor(
        [cfg.POSITIVE_WEIGHT],
        dtype=torch.float32,
        device=device,
    )
)


# ============================================================
# Optimizer and scheduler
# ============================================================

# Train only the CNN head.
# The Gabor front end remains fixed.
optimizer = torch.optim.AdamW(
    model.head.parameters(),
    lr=cfg.LEARNING_RATE,
    weight_decay=cfg.WEIGHT_DECAY,
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=cfg.SCHEDULER_FACTOR,
    patience=cfg.SCHEDULER_PATIENCE,
    min_lr=cfg.MIN_LEARNING_RATE,
)


# ============================================================
# Checkpoint setup
# ============================================================

cfg.CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

best_val_loss = float("inf")

print(f"Checkpoint path: {cfg.CHECKPOINT_PATH}")


# ============================================================
# Training loop
# ============================================================

for epoch in range(cfg.EPOCHS):

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    model.train()
    train_loss_sum = 0.0

    for images, targets in train_loader:

        images = images.to(device)
        targets = targets.to(device)

        # Apply the same spatial augmentation to the image
        # and its ground-truth boundary map.

        if random.random() < 0.5:
            images = torch.flip(images, dims=[3])
            targets = torch.flip(targets, dims=[3])

        if random.random() < 0.5:
            images = torch.flip(images, dims=[2])
            targets = torch.flip(targets, dims=[2])

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)

        loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item() * images.size(0)

    train_loss = train_loss_sum / len(train_set)

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    model.eval()
    val_loss_sum = 0.0

    with torch.no_grad():

        for images, targets in val_loader:

            images = images.to(device)
            targets = targets.to(device)

            logits = model(images)

            loss = criterion(logits, targets)

            val_loss_sum += loss.item() * images.size(0)

    val_loss = val_loss_sum / len(val_set)

    # Update learning rate based on validation loss.
    scheduler.step(val_loss)

    current_lr = optimizer.param_groups[0]["lr"]

    print(
        f"Epoch {epoch + 1:02d}/{cfg.EPOCHS} | "
        f"Train loss: {train_loss:.5f} | "
        f"Val loss: {val_loss:.5f} | "
        f"LR: {current_lr:.2e}"
    )

    # --------------------------------------------------------
    # Save the best model
    # --------------------------------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),

            # Model configuration
            "sigma": cfg.GABOR_SIGMA,
            "wavelength": cfg.GABOR_WAVELENGTH,
            "gamma": cfg.GABOR_GAMMA,
            "orientations": cfg.ORIENTATIONS,
            "hidden_channels": cfg.HIDDEN_CHANNELS,

            # Input configuration
            "input_size": (
                cfg.IMAGE_HEIGHT,
                cfg.IMAGE_WIDTH,
            ),

            # Training configuration
            "best_val_loss": best_val_loss,
            "learning_rate": cfg.LEARNING_RATE,
            "positive_weight": cfg.POSITIVE_WEIGHT,
            "random_seed": cfg.RANDOM_SEED,
        }

        torch.save(
            checkpoint,
            cfg.CHECKPOINT_PATH,
        )

        print(
            f"  Saved best checkpoint: "
            f"{cfg.CHECKPOINT_PATH}"
        )


# ============================================================
# Training summary
# ============================================================

print("\nTraining complete.")
print(f"Best validation loss: {best_val_loss:.5f}")
print(f"Best checkpoint: {cfg.CHECKPOINT_PATH}")
