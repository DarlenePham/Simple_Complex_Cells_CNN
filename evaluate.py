from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

import config
from dataset import make_consensus_boundary
from Models.complex_cell_model import SimpleComplexEdgeNet

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = Path(getattr(config, "DATASET_ROOT", PROJECT_ROOT / "BSDS300"))
ANNOTATION_DIR = Path(
    getattr(config, "ANNOTATION_DIR", DATASET_ROOT / "human" / "gray")
)
CHECKPOINT_PATH = Path(
    getattr(
        config,
        "CHECKPOINT_PATH",
        PROJECT_ROOT / "checkpoints" / "simple_complex_best.pth",
    )
)
IMAGE_HEIGHT = int(getattr(config, "IMAGE_HEIGHT", 256))
IMAGE_WIDTH = int(getattr(config, "IMAGE_WIDTH", 384))

# Back-fill a minimal configuration expected by the dataset helpers.
setattr(config, "ANNOTATION_DIR", ANNOTATION_DIR)
setattr(config, "DATASET_ROOT", DATASET_ROOT)
setattr(config, "TARGET_MODE", getattr(config, "TARGET_MODE", "soft"))
setattr(config, "BOUNDARY_CONNECTIVITY", getattr(config, "BOUNDARY_CONNECTIVITY", 1))
setattr(config, "GRAYSCALE", getattr(config, "GRAYSCALE", True))
setattr(
    config, "IMAGE_SIZE", getattr(config, "IMAGE_SIZE", (IMAGE_HEIGHT, IMAGE_WIDTH))
)


def _resolve_image_path(image_spec: str | Path) -> Path:
    """Resolve an image path from a file path, image id, or stem."""
    candidate = Path(image_spec)
    if candidate.exists():
        return candidate

    search_roots = [
        DATASET_ROOT / "images" / "train",
        DATASET_ROOT / "images" / "test",
        PROJECT_ROOT,
    ]

    if candidate.suffix:
        patterns = [candidate]
    else:
        patterns = [
            candidate.with_suffix(".jpg"),
            candidate.with_suffix(".png"),
            candidate.with_suffix(".jpeg"),
        ]

    for root in search_roots:
        for pattern in patterns:
            if pattern.is_absolute():
                continue
            matches = sorted(root.rglob(pattern.name))
            for match in matches:
                if match.stem == candidate.stem:
                    return match

        if candidate.suffix == "":
            for match in sorted(root.rglob("*")):
                if match.stem == candidate.name and match.suffix.lower() in {
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".bmp",
                    ".tif",
                    ".tiff",
                }:
                    return match

    # If the user gave an image id like 101085, search by stem anywhere under the dataset root.
    stem = Path(str(candidate)).stem
    if stem and stem != ".":
        for root in search_roots:
            matches = sorted(root.rglob("*"))
            for match in matches:
                if (
                    match.suffix.lower()
                    in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
                    and match.stem == stem
                ):
                    return match

    raise FileNotFoundError(
        f"Could not resolve image '{image_spec}' under {DATASET_ROOT}."
    )


def _rescale_image_to_uint8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.dtype.kind != "u" and image.max() > 1.0:
        image = image / 255.0
    image = np.clip(image, 0.0, 1.0)
    return np.clip(image * 255.0, 0, 255).astype(np.uint8)


def _resize_for_model(
    image: np.ndarray, height: int = IMAGE_HEIGHT, width: int = IMAGE_WIDTH
) -> np.ndarray:
    if image.ndim == 2:
        image = image.astype(np.float32)
    elif image.shape[-1] == 3:
        image = image.mean(axis=-1)
    elif image.shape[0] == 3:
        image = image.transpose(1, 2, 0).mean(axis=-1)

    pil_image = Image.fromarray(_rescale_image_to_uint8(image), mode="L")
    resized = pil_image.resize((width, height), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32) / 255.0


def _resize_probability_map(
    probability: np.ndarray, target_shape: tuple[int, int]
) -> np.ndarray:
    height, width = target_shape
    if probability.shape == target_shape:
        return probability.astype(np.float32)

    pil_image = Image.fromarray(probability.astype(np.float32), mode="F")
    resized = pil_image.resize((width, height), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32)


def load_model(
    checkpoint_path: str | Path = CHECKPOINT_PATH, device: Optional[torch.device] = None
) -> tuple[torch.nn.Module, torch.device]:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = SimpleComplexEdgeNet(
        sigma=checkpoint.get("sigma", 2.5),
        wavelength=checkpoint.get("wavelength", 5.0),
        gamma=checkpoint.get("gamma", 0.5),
        orientations=checkpoint.get("orientations", None),
        hidden_channels=checkpoint.get("hidden_channels", 32),
    ).to(device)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, device


def predict_boundary_probability(
    image: np.ndarray,
    model: torch.nn.Module,
    device: Optional[torch.device] = None,
    original_shape: Optional[tuple[int, int]] = None,
) -> np.ndarray:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if original_shape is None:
        original_shape = image.shape[:2]

    resized_image = _resize_for_model(image)
    image_tensor = torch.from_numpy(resized_image).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.sigmoid(logits)[0, 0].cpu().numpy()

    return _resize_probability_map(probabilities, original_shape)


def precision_recall_f1(
    prediction: np.ndarray,
    target: np.ndarray,
    threshold: float = 0.5,
    threshold_percentile: float | None = None,
) -> dict:
    prediction = np.asarray(prediction, dtype=np.float32)
    target = np.asarray(target)

    # Calculate a threshold from the prediction map if requested.
    if threshold_percentile is not None:
        threshold = float(np.percentile(prediction, threshold_percentile))

    # Binarize prediction and human ground truth.
    prediction_binary = prediction >= threshold
    target_binary = target >= 0.5

    tp = np.logical_and(prediction_binary, target_binary).sum()
    fp = np.logical_and(prediction_binary, ~target_binary).sum()
    fn = np.logical_and(~prediction_binary, target_binary).sum()

    precision = tp / float(tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / float(tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "threshold": float(threshold),
        "threshold_percentile": threshold_percentile,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
    }


def normalize_map(map_array: np.ndarray) -> np.ndarray:
    array = np.asarray(map_array, dtype=np.float32)
    min_value = float(array.min())
    max_value = float(array.max())
    if max_value <= min_value:
        return np.zeros_like(array, dtype=np.float32)
    return (array - min_value) / (max_value - min_value + 1e-8)


def extract_simple_and_complex_maps(
    model: torch.nn.Module, image: np.ndarray, device: torch.device
) -> dict:
    resized_image = _resize_for_model(image)
    image_tensor = torch.from_numpy(resized_image).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        features = model(image_tensor, return_features=True)

    simple = features["simple"][0].cpu().numpy()
    complex = features["complex"][0].cpu().numpy()

    simple_map = np.abs(simple).mean(axis=0)
    complex_map = complex.mean(axis=0)

    simple_map = _resize_probability_map(normalize_map(simple_map), image.shape[:2])
    complex_map = _resize_probability_map(normalize_map(complex_map), image.shape[:2])

    return {
        "simple": simple_map,
        "complex": complex_map,
        "trained": predict_boundary_probability(
            image, model, device=device, original_shape=image.shape[:2]
        ),
    }


def evaluate_image(
    image_spec: str | Path,
    annotation_root: str | Path = ANNOTATION_DIR,
    checkpoint_path: str | Path = CHECKPOINT_PATH,
    threshold: float = 0.5,
) -> dict:
    """Compare a BSDS image against the human segmentation using simple, complex, and trained features."""
    image_path = _resolve_image_path(image_spec)
    image = np.asarray(Image.open(image_path).convert("L"), dtype=np.float32) / 255.0

    seg_paths = sorted(Path(annotation_root).glob(f"*/{image_path.stem}.seg"))
    if not seg_paths:
        raise FileNotFoundError(
            f"No BSDS human annotations found for '{image_path.name}' under {annotation_root}."
        )

    target = make_consensus_boundary(seg_paths, target_mode="soft")
    if image.shape[:2] != target.shape:
        target = _resize_probability_map(target, image.shape[:2])

    metrics = {
        "image": str(image_path),
        "image_id": image_path.stem,
        "annotation_count": len(seg_paths),
        "original_shape": image.shape[:2],
        "target_shape": target.shape,
        "human_consensus_mean": float(target.mean()),
        "human_consensus_max": float(target.max()),
    }

    if checkpoint_path and Path(checkpoint_path).exists():
        model, device = load_model(checkpoint_path=checkpoint_path)
        feature_maps = extract_simple_and_complex_maps(model, image, device)

        metrics["simple_cell"] = feature_maps["simple"]
        metrics["complex_cell"] = feature_maps["complex"]
        metrics["trained_boundary"] = feature_maps["trained"]

        metrics["simple_metrics"] = precision_recall_f1(
            feature_maps["simple"], target, threshold_percentile=90
        )
        metrics["complex_metrics"] = precision_recall_f1(
            feature_maps["complex"], target, threshold_percentile=90
        )
        metrics["trained_metrics"] = precision_recall_f1(
            feature_maps["trained"], target, threshold_percentile=90
        )

    metrics["human_metrics"] = precision_recall_f1(target, target, threshold=0.5)
    metrics["threshold"] = float(threshold)
    return metrics


def compare_image(
    image_spec: str | Path,
    annotation_root: str | Path = ANNOTATION_DIR,
    checkpoint_path: str | Path = CHECKPOINT_PATH,
    threshold: float = 0.5,
) -> dict:
    return evaluate_image(
        image_spec=image_spec,
        annotation_root=annotation_root,
        checkpoint_path=checkpoint_path,
        threshold=threshold,
    )


def _format_metrics(metrics: dict) -> str:
    lines = [
        f"image: {metrics['image']}",
        f"image_id: {metrics['image_id']}",
        f"human annotations: {metrics['annotation_count']}",
        f"target shape: {metrics['target_shape']}",
    ]

    sections = [
        ("simple cell", metrics.get("simple_metrics")),
        ("complex cell", metrics.get("complex_metrics")),
        ("trained boundary", metrics.get("trained_metrics")),
    ]

    for label, stats in sections:
        if stats is None:
            lines.append(f"{label}: not available")
            continue

        lines.append(
            f"{label} -> precision: {stats['precision']:.4f} | recall: {stats['recall']:.4f} | f1: {stats['f1']:.4f}"
        )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate simple-cell response, complex-cell response, and trained boundary prediction against BSDS human segmentation."
    )
    parser.add_argument(
        "image",
        nargs="?",
        default="101085",
        help="Image path or BSDS image id to evaluate.",
    )
    parser.add_argument(
        "--image-id", dest="image_id", help="BSDS image id (e.g. 101085)."
    )
    parser.add_argument(
        "--annotation-root",
        type=str,
        default=str(ANNOTATION_DIR),
        help="Folder containing BSDS human .seg files.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(CHECKPOINT_PATH),
        help="Path to the model checkpoint to compare against BSDS edges.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold used for binarizing each response map.",
    )
    args = parser.parse_args()

    image_spec = args.image_id if args.image_id else args.image
    results = evaluate_image(
        image_spec=image_spec,
        annotation_root=args.annotation_root,
        checkpoint_path=args.checkpoint,
        threshold=args.threshold,
    )
    print(_format_metrics(results))


if __name__ == "__main__":
    main()
