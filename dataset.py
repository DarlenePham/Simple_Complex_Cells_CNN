from pathlib import Path
import numpy as np
from PIL import Image
from skimage.segmentation import find_boundaries
from skimage.transform import resize
import config as config


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# ============================================================
# Image loading
# ============================================================

def load_image(image_path):
    """Load an image as a normalized NumPy array."""

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with Image.open(image_path) as img:
        img = img.convert("L" if config.GRAYSCALE else "RGB")
        image = np.asarray(img, dtype=np.float32) / 255.0

    if config.IMAGE_SIZE is not None:
        image = resize(
            image,
            config.IMAGE_SIZE,
            order=1,
            preserve_range=True,
            anti_aliasing=True,
        ).astype(np.float32)

    return image


# ============================================================
# .seg file parsing
# ============================================================

def load_segmentation(seg_path):
    """
    Reconstruct a segmentation label map from a BSDS .seg file.

    Expected data rows:
        segment_id row_index start_x end_x

    Pixel coordinates are treated as inclusive at both ends.

    Returns
    -------
    labels : np.ndarray
        Integer array of shape (height, width).
        Each pixel contains its segment ID.
    """

    seg_path = Path(seg_path)

    if not seg_path.exists():
        raise FileNotFoundError(f"Segmentation file not found: {seg_path}")

    width = None
    height = None
    expected_segments = None
    in_data = False
    runs = []

    with seg_path.open("r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()

            if not line:
                continue

            if not in_data:
                parts = line.split()
                key = parts[0].lower()

                if key == "width":
                    width = int(parts[1])

                elif key == "height":
                    height = int(parts[1])

                elif key == "segments":
                    expected_segments = int(parts[1])

                elif key == "data":
                    in_data = True

                continue

            # Data section: segment_id y x_start x_end
            parts = line.split()

            if len(parts) != 4:
                raise ValueError(
                    f"Invalid data row in {seg_path}: {line}"
                )

            segment_id, y, x_start, x_end = map(int, parts)
            runs.append((segment_id, y, x_start, x_end))

    if width is None or height is None:
        raise ValueError(
            f"Missing width or height in {seg_path}"
        )

    if not in_data:
        raise ValueError(f"Missing data section in {seg_path}")

    labels = np.full((height, width), -1, dtype=np.int32)

    for segment_id, y, x_start, x_end in runs:
        if not (0 <= y < height):
            raise ValueError(
                f"Invalid row index {y} in {seg_path}"
            )

        if not (0 <= x_start <= x_end < width):
            raise ValueError(
                f"Invalid x range {x_start}:{x_end} in {seg_path}"
            )

        if np.any(labels[y, x_start:x_end + 1] != -1):
            raise ValueError(
                f"Overlapping pixel runs in {seg_path}, row {y}"
            )

        # End coordinate is inclusive.
        labels[y, x_start:x_end + 1] = segment_id

    if np.any(labels == -1):
        missing = int(np.count_nonzero(labels == -1))
        raise ValueError(
            f"{seg_path} leaves {missing} pixels unassigned"
        )

    if expected_segments is not None:
        actual_segments = len(np.unique(labels))

        if actual_segments != expected_segments:
            raise ValueError(
                f"Segment count mismatch in {seg_path}: "
                f"header={expected_segments}, actual={actual_segments}"
            )

    return labels


# ============================================================
# Boundary extraction
# ============================================================

def segmentation_to_boundary(labels):
    """
    Convert a region-label map to a binary boundary map.
    """

    boundary = find_boundaries(
        labels,
        connectivity=config.BOUNDARY_CONNECTIVITY,
        mode="thick",
        background=0,
    )

    return boundary.astype(np.float32)


def load_annotator_boundaries(seg_paths):
    """
    Load all .seg files for an image.

    Returns
    -------
    boundaries : np.ndarray
        Shape (number_of_annotators, height, width).
    """

    boundaries = []

    for seg_path in seg_paths:
        labels = load_segmentation(seg_path)
        boundary = segmentation_to_boundary(labels)
        boundaries.append(boundary)

    if not boundaries:
        raise ValueError("No segmentation annotations provided.")

    shapes = {boundary.shape for boundary in boundaries}

    if len(shapes) != 1:
        raise ValueError(
            f"Annotation dimensions do not match: {shapes}"
        )

    return np.stack(boundaries, axis=0)


# ============================================================
# Consensus boundary target
# ============================================================

def make_consensus_boundary(seg_paths, target_mode=None):
    """
    Combine the boundaries from multiple human annotators.

    soft:
        Each pixel stores the fraction of annotators who
        marked it as a boundary.

    binary:
        Each pixel is 1 if any annotator marked a boundary.
    """

    if target_mode is None:
        target_mode = config.TARGET_MODE

    boundaries = load_annotator_boundaries(seg_paths)

    soft_target = boundaries.mean(axis=0).astype(np.float32)

    if target_mode == "soft":
        return soft_target

    if target_mode == "binary":
        return (soft_target > 0).astype(np.float32)

    raise ValueError(
        f"Unknown target mode: {target_mode}"
    )


# ============================================================
# Dataset indexing
# ============================================================

def list_images(image_dir):
    """List image files in one directory."""

    image_dir = Path(image_dir)

    if not image_dir.is_dir():
        raise FileNotFoundError(
            f"Image directory does not exist: {image_dir}"
        )

    return sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def find_annotations(image_path, annotation_root=None):
    """
    Find every annotator's .seg file for an image.

    For image 22093.jpg, searches for paths such as:

        human/gray/1102/22093.seg
        human/gray/1103/22093.seg
        human/gray/1104/22093.seg

    Missing annotations are skipped, but at least one
    annotation must exist.
    """

    if annotation_root is None:
        annotation_root = config.ANNOTATION_DIR

    image_path = Path(image_path)
    annotation_root = Path(annotation_root)

    if not annotation_root.is_dir():
        raise FileNotFoundError(
            f"Annotation directory does not exist: {annotation_root}"
        )

    seg_paths = sorted(
        annotation_root.glob(f"*/{image_path.stem}.seg")
    )

    if not seg_paths:
        raise FileNotFoundError(
            f"No .seg annotations found for {image_path.name} "
            f"under {annotation_root}"
        )

    return seg_paths


# ============================================================
# Dataset class
# ============================================================

class BSDSDataset:
    """
    Lightweight BSDS300 dataset.

    Example
    -------
    dataset = BSDSDataset(
        config.TRAIN_IMAGE_DIR,
        config.ANNOTATION_DIR,
    )

    image, target, image_id = dataset[0]
    """

    def __init__(
        self,
        image_dir,
        annotation_dir=None,
        target_mode=None,
    ):
        self.image_dir = Path(image_dir)

        if annotation_dir is None:
            annotation_dir = config.ANNOTATION_DIR

        self.annotation_dir = Path(annotation_dir)
        self.target_mode = target_mode or config.TARGET_MODE

        self.image_paths = list_images(self.image_dir)

        if not self.image_paths:
            raise ValueError(
                f"No images found in {self.image_dir}"
            )

        # Match each image to all available human annotations.
        self.samples = []

        for image_path in self.image_paths:
            seg_paths = find_annotations(
                image_path,
                self.annotation_dir,
            )

            self.samples.append((image_path, seg_paths))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, seg_paths = self.samples[index]

        image = load_image(image_path)

        target = make_consensus_boundary(
            seg_paths,
            target_mode=self.target_mode,
        )

        if image.shape[:2] != target.shape:
            if config.IMAGE_SIZE is None:
                raise ValueError(
                    f"Image/target shape mismatch for "
                    f"{image_path.name}: "
                    f"{image.shape[:2]} vs {target.shape}"
                )

            target = resize(
                target,
                image.shape[:2],
                order=1,
                preserve_range=True,
                anti_aliasing=False,
            ).astype(np.float32)

        return image, target, image_path.stem


# ============================================================
# Standalone test
# ============================================================

if __name__ == "__main__":
    dataset = BSDSDataset(
        config.TRAIN_IMAGE_DIR,
        config.ANNOTATION_DIR,
    )

    print("Number of images:", len(dataset))

    image, target, image_id = dataset[0]

    print("Image ID:", image_id)
    print("Image shape:", image.shape)
    print("Target shape:", target.shape)
    print("Target range:", target.min(), target.max())
    print("Target mean:", target.mean())

    image_path, seg_paths = dataset.samples[0]

    print("Image path:", image_path)
    print("Number of human annotations:", len(seg_paths))
    print("Example annotation:", seg_paths[0])
