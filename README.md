# Simple and Complex Cells for Boundary Detection

This project explores image-boundary detection using a fixed Gabor simple/complex-cell front end and a trainable convolutional head. The training targets are boundary maps derived from human segmentations in the Berkeley Segmentation Dataset (BSDS300).

## How it works

1. A grayscale image is resized to `256 x 384` by default.
2. A bank of Gabor filters at eight orientations produces two phase responses per orientation (simple-cell responses).
3. Each pair of phase responses is combined into a phase-invariant energy map (complex-cell response).
4. The simple and complex responses are concatenated and passed to a CNN head, which predicts per-pixel boundary logits.
5. Training compares those logits to a soft consensus of the available human boundary annotations using weighted binary cross-entropy.

The Gabor filters are fixed; training updates the CNN head. Evaluation compares the simple-cell response, complex-cell response, and trained boundary prediction with human-consensus boundaries.

## Project layout

```text
.
├── BSDS300/                    # Local BSDS300 data
│   ├── images/
│   │   ├── train/
│   │   └── test/
│   └── human/
│       └── gray/<annotator>/
├── Models/
│   ├── bsds_dataset.py         # BSDS annotations and PyTorch dataset
│   ├── complex_cell_model.py   # Gabor front end and trainable edge network
│   └── gabor_filter.py         # Gabor kernel generation
├── checkpoints/                # Saved model checkpoint
├── config.py                   # Dataset, model, and training settings
├── train_model.py              # Training entry point
└── evaluate.py                 # Single-image BSDS evaluation
```

## Requirements

Use Python 3.10 or later. Install PyTorch using the instructions for your platform and desired CPU/CUDA setup from [pytorch.org](https://pytorch.org/get-started/locally/), then install the remaining runtime packages:

```powershell
python -m pip install numpy pillow scipy matplotlib
```

Download the BSDS300 image dataset here: https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/

If you plan to use other dataset, place them in the root just like example in Project Layout and modify the DATASET_ROOT in `config.py`.

The project currently does not include a dependency lockfile or `requirements.txt`.

## Dataset setup

Place the BSDS300 files in the `BSDS300` directory at the project root (or update `DATASET_ROOT` in `config.py`). Training and evaluation expect image files and human `.seg` annotations with matching image IDs. The default annotation type is `gray`.

```text
BSDS300/
├── images/
│   ├── train/
│   └── test/
└── human/
    └── gray/
        ├── 1102/
        │   └── <image-id>.seg
        └── 1103/
            └── <image-id>.seg
```

An image is used only when matching human annotation files are present. For evaluation, the image and its annotations must share the same filename stem (for example, `101085.jpg` and `101085.seg` in one or more annotator directories).

## Train

From the project root, run:

```powershell
python train_model.py
```

Training uses the `train` images, splitting off the fraction in `VALIDATION_FRACTION` for validation. The best validation-loss checkpoint is saved to:

```text
checkpoints/simple_complex_best.pth
```

Common settings are in `config.py`, including image dimensions, batch size, number of epochs, learning rate, Gabor parameters, and checkpoint path. The default device is CUDA when available, otherwise CPU.

## Evaluate an image

After training (or placing a compatible checkpoint at the configured checkpoint path), evaluate a BSDS image by ID:

```powershell
python evaluate.py --image-id 101085
```

You can also pass an image path:

```powershell
python evaluate.py "C:\path\to\101085.jpg"
```

Optional arguments:

```text
--annotation-root PATH   Root containing annotator subdirectories and .seg files
--checkpoint PATH        Model checkpoint to load
--threshold FLOAT        Threshold option (default: 0.5; see note below)
```

The evaluator reports precision, recall, and F1 for simple-cell response, complex-cell response, and trained boundary prediction. It currently thresholds each response map at its own 90th percentile for these metrics; the human consensus target is binarized at `0.5`. Although `--threshold` is accepted, the current percentile-based metric calls do not use that argument. If no checkpoint exists at the requested path, the evaluator can still build the human consensus, but model metrics are reported as unavailable.

## Configuration and notes

- The dataset root defaults to `BSDS300/` next to `config.py`.
- Training uses `IMAGE_HEIGHT = 256` and `IMAGE_WIDTH = 384` by default.
- Human annotations are averaged to make a soft target: a value near 1 means annotators agree that a pixel is a boundary.
- `train_model.py` trains and validates on the training split; it does not automatically report test-split metrics.