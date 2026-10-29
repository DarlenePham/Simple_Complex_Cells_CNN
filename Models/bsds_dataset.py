
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


def read_seg_file(path):
    """Decode a Berkeley ASCII compressed-row .seg file."""
    width = height = None
    data_started = False
    segments = []

    with open(path, "r", encoding="ascii", errors="replace") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            if not data_started:
                if line.lower() == "data":
                    data_started = True
                    continue

                parts = line.split()

                if len(parts) >= 2:
                    key = parts[0].lower()

                    if key == "width":
                        width = int(parts[1])
                    elif key == "height":
                        height = int(parts[1])
                    elif key == "format":
                        fmt = " ".join(parts[1:]).lower()
                        if fmt not in ("ascii cr", "ascii"):
                            raise ValueError(
                                f"Unsupported .seg format: {fmt}"
                            )

            else:
                values = line.split()

                if len(values) != 4:
                    continue

                segment_id, row, col_start, col_end = map(
                    int, values
                )
                segments.append(
                    (segment_id, row, col_start, col_end)
                )

    if width is None or height is None or not data_started:
        raise ValueError(f"Invalid .seg header: {path}")

    labels = np.full((height, width), -1, dtype=np.int32)

    for segment_id, row, col_start, col_end in segments:
        if not (0 <= row < height):
            continue

        col_start = max(0, col_start)
        col_end = min(width - 1, col_end)

        if col_start <= col_end:
            labels[row, col_start:col_end + 1] = segment_id

    if np.any(labels < 0):
        raise ValueError(
            f"Incomplete segmentation in {path}: "
            f"{np.count_nonzero(labels < 0)} pixels missing"
        )

    return labels


def segmentation_to_boundary(labels):
    """Mark pixels adjacent to different segment IDs."""
    boundary = np.zeros(labels.shape, dtype=np.float32)

    horizontal = labels[:, 1:] != labels[:, :-1]
    vertical = labels[1:, :] != labels[:-1, :]

    boundary[:, 1:] = np.maximum(boundary[:, 1:], horizontal)
    boundary[:, :-1] = np.maximum(boundary[:, :-1], horizontal)

    boundary[1:, :] = np.maximum(boundary[1:, :], vertical)
    boundary[:-1, :] = np.maximum(boundary[:-1, :], vertical)

    return boundary


class BSDSBoundaryDataset(Dataset):
    """
    Build a soft boundary target by averaging human segmentations.

    Expected root:
        BSDS300/
            images/train/
            images/test/
            human/gray/<annotator>/*.seg
    """

    def __init__(
        self,
        root,
        split="train",
        size=(256, 384),
        annotation_type="gray"
    ):
        self.root = Path(root)
        self.split = split
        self.size = size  # (height, width)

        image_dir = self.root / "images" / split
        if not image_dir.exists():
            raise FileNotFoundError(
                f"Image directory not found: {image_dir}"
            )

        extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        image_paths = [
            p for p in image_dir.rglob("*")
            if p.suffix.lower() in extensions
        ]

        # Index all human segmentations by image ID.
        annotation_root = self.root / "human" / annotation_type
        if not annotation_root.exists():
            raise FileNotFoundError(
                f"Annotation directory not found: {annotation_root}"
            )

        annotations = {}
        for path in annotation_root.rglob("*.seg"):
            annotations.setdefault(path.stem, []).append(path)

        self.samples = []

        for image_path in sorted(image_paths):
            seg_paths = annotations.get(image_path.stem, [])

            if seg_paths:
                self.samples.append((image_path, sorted(seg_paths)))

        if not self.samples:
            raise RuntimeError(
                f"No matching images and .seg annotations found. "
                f"Images: {image_dir}, annotations: {annotation_root}"
            )

        print(
            f"Loaded {len(self.samples)} {split} images "
            f"with human annotations."
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, seg_paths = self.samples[index]

        # Grayscale image in [0, 1].
        image = Image.open(image_path).convert("L")
        image = image.resize(
            (self.size[1], self.size[0]),
            Image.Resampling.BILINEAR
        )
        image = np.asarray(image, dtype=np.float32) / 255.0

        boundary_maps = []

        for seg_path in seg_paths:
            labels = read_seg_file(seg_path)

            # Resize segment IDs with nearest-neighbor interpolation.
            label_image = Image.fromarray(labels)
            label_image = label_image.resize(
                (self.size[1], self.size[0]),
                Image.Resampling.NEAREST
            )

            labels_resized = np.asarray(label_image)
            boundary_maps.append(
                segmentation_to_boundary(labels_resized)
            )

        # Soft consensus: values near 1 indicate strong annotator agreement.
        target = np.mean(boundary_maps, axis=0).astype(np.float32)

        image = torch.from_numpy(image[None, :, :])
        target = torch.from_numpy(target[None, :, :])

        return image, target
