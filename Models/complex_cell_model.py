
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from Models.gabor_filter import gabor


class GaborSimpleComplexCells(nn.Module):
    """Fixed Gabor simple/complex cell feature extractor."""

    def __init__(
        self,
        sigma=2.5,
        wavelength=5.0,
        gamma=0.5,
        orientations=None
    ):
        super().__init__()

        if orientations is None:
            orientations = np.arange(0, 180, 22.5)

        self.orientations = np.asarray(orientations, dtype=float)

        
        kernels = []

        for angle in self.orientations:
            for phase in (0.0, np.pi / 2):
                kernel = gabor(
                    sigma=sigma,
                    theta=np.deg2rad(angle),
                    Lambda=wavelength,
                    psi=phase,
                    gamma=gamma,
                ).astype(np.float32)

                # Normalize the filter
                kernel -= kernel.mean()
                kernel /= np.abs(kernel).sum() + 1e-8

                kernels.append(kernel)

        # Find the largest kernel dimensions
        max_height = max(k.shape[0] for k in kernels)
        max_width = max(k.shape[1] for k in kernels)

        # Pad every kernel to the same dimensions, keeping it centered
        padded_kernels = []

        for kernel in kernels:
            height, width = kernel.shape

            pad_top = (max_height - height) // 2
            pad_bottom = max_height - height - pad_top
            pad_left = (max_width - width) // 2
            pad_right = max_width - width - pad_left

            padded = np.pad(
                kernel,
                ((pad_top, pad_bottom), (pad_left, pad_right)),
                mode="constant",
                constant_values=0,
            )

            padded_kernels.append(padded)

        # Shape: [num_orientations * 2, 1, max_height, max_width]
        weights = np.stack(padded_kernels)[:, None, :, :]

        self.register_buffer(
            "gabor_weights",
            torch.from_numpy(weights),
        )

        self.num_orientations = len(self.orientations)


    def forward(self, image):
        """
        image: [B, 1, H, W], grayscale in [0, 1]

        Returns:
            simple:  [B, 2*O, H, W], signed phase responses
            complex: [B, O, H, W], quadrature energy responses
        """
        responses = F.conv2d(
            image,
            self.gabor_weights,
            padding="same"
        )

        # Filter order: phase 0, phase pi/2, repeated per orientation.
        simple = responses

        paired = responses.reshape(
            responses.shape[0],
            self.num_orientations,
            2,
            responses.shape[-2],
            responses.shape[-1]
        )

        complex_response = torch.sqrt(
            paired[:, :, 0].square()
            + paired[:, :, 1].square()
            + 1e-8
        )

        return simple, complex_response


class SimpleComplexEdgeNet(nn.Module):
    """
    Gabor simple/complex cells followed by a trainable CNN
    that predicts per-pixel boundary logits.
    """

    def __init__(
        self,
        sigma=2.5,
        wavelength=5.0,
        gamma=0.5,
        orientations=None,
        hidden_channels=32
    ):
        super().__init__()

        self.cells = GaborSimpleComplexCells(
            sigma=sigma,
            wavelength=wavelength,
            gamma=gamma,
            orientations=orientations
        )

        num_orientations = self.cells.num_orientations
        input_channels = 3 * num_orientations

        self.head = nn.Sequential(
            nn.Conv2d(
                input_channels, hidden_channels,
                kernel_size=3, padding=1
            ),
            nn.GroupNorm(4, hidden_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                hidden_channels, hidden_channels,
                kernel_size=3, padding=1
            ),
            nn.GroupNorm(4, hidden_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                hidden_channels, hidden_channels // 2,
                kernel_size=3, padding=1
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_channels // 2, 1, kernel_size=1)
        )

    def forward(self, image, return_features=False):
        simple, complex_response = self.cells(image)

        # Concatenate signed simple-cell responses and
        # phase-invariant complex-cell energy responses.
        features = torch.cat(
            [simple, complex_response],
            dim=1
        )

        logits = self.head(features)

        if return_features:
            return {
                "logits": logits,
                "simple": simple,
                "complex": complex_response
            }

        return logits
