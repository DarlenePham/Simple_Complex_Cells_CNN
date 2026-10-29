
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate2d

def gabor(sigma, theta, Lambda, psi, gamma):
    """
    Generate a 2D Gabor filter.

    Parameters
    ----------
    sigma : float
        Gaussian envelope scale.
    theta : float
        Preferred orientation in radians.
    Lambda : float
        Sinusoidal wavelength.
    psi : float
        Phase offset in radians.
    gamma : float
        Spatial aspect ratio.

    Returns
    -------
    kernel : ndarray
        2D Gabor receptive field with positive and negative weights.
    """
    if sigma <= 0 or Lambda <= 0 or gamma <= 0:
        raise ValueError("sigma, Lambda, and gamma must be positive.")

    sigma_x = float(sigma)
    sigma_y = float(sigma) / gamma

    # Bounding box: three standard deviations
    nstds = 3

    xmax = int(np.ceil(max(
        1,
        abs(nstds * sigma_x * np.cos(theta)),
        abs(nstds * sigma_y * np.sin(theta))
    )))

    ymax = int(np.ceil(max(
        1,
        abs(nstds * sigma_x * np.sin(theta)),
        abs(nstds * sigma_y * np.cos(theta))
    )))

    # Correct row/column coordinate ordering
    y, x = np.mgrid[-ymax:ymax + 1, -xmax:xmax + 1]

    # Rotate coordinates
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)

    # Gaussian envelope multiplied by a cosine wave
    kernel = np.exp(
        -0.5 * (
            x_theta**2 / sigma_x**2
            + y_theta**2 / sigma_y**2
        )
    ) * np.cos(2 * np.pi * x_theta / Lambda + psi)

    return kernel

def _normalize_kernel(kernel):
    """
    Remove the DC component and normalize the absolute weight sum.
    """
    kernel = kernel.astype(np.float64, copy=True)
    kernel -= np.mean(kernel)

    weight_sum = np.sum(np.abs(kernel))

    if weight_sum > 1e-12:
        kernel /= weight_sum

    return kernel

def compute_response_map(image, sigma, theta, Lambda, psi, gamma):
    """
    Apply one normalized Gabor filter to a grayscale image.

    Returns a response map with the same height and width as the input.
    Positive and negative responses are preserved.
    """
    image = np.asarray(image, dtype=np.float64)

    if image.ndim != 2:
        raise ValueError("Input image must be a 2D grayscale image.")

    if image.size == 0:
        raise ValueError("Input image cannot be empty.")

    kernel = gabor(sigma, theta, Lambda, psi, gamma)
    kernel = _normalize_kernel(kernel)

    return correlate2d(
        image,
        kernel,
        mode="same",
        boundary="symm"
    )

def compute_simple_cell_response(
    image,
    sigma=3.0,
    Lambda=6.0,
    psi=0.0,
    gamma=0.5,
    orientations=None,
    threshold_percentile=90
):
    """
    Run a bank of oriented Gabor filters to approximate simple cells.

    Parameters
    ----------
    image : ndarray
        2D grayscale input image.
    sigma : float
        Gaussian envelope scale.
    Lambda : float
        Wavelength of the sinusoidal component.
    psi : float
        Phase offset.
    gamma : float
        Spatial aspect ratio.
    orientations : sequence of float, optional
        Filter orientations in degrees.
        Defaults to 0, 22.5, ..., 157.5 degrees.
    threshold_percentile : float
        Percentile used to binarize the combined edge response.

    Returns
    -------
    response_maps : dict
        Signed response maps indexed by orientation in degrees.
    edge_response : ndarray
        Normalized maximum absolute response across orientations.
    binary_response : ndarray
        Binary edge map, with 1 indicating a strong response.
    receptive_fields : dict
        Normalized Gabor kernels indexed by orientation in degrees.
    """
    image = np.asarray(image, dtype=np.float64)

    if image.ndim != 2:
        raise ValueError("Input image must be a 2D grayscale image.")

    if image.size == 0:
        raise ValueError("Input image cannot be empty.")

    if not 0 <= threshold_percentile <= 100:
        raise ValueError("threshold_percentile must be between 0 and 100.")

    if orientations is None:
        orientations = np.arange(0, 180, 22.5)

    orientations = np.asarray(orientations, dtype=float)

    if orientations.ndim != 1 or orientations.size == 0:
        raise ValueError("orientations must be a non-empty sequence.")

    if not np.all(np.isfinite(orientations)):
        raise ValueError("orientations must contain finite values.")

    # Normalize orientation values to [0, 180)
    orientations = np.mod(orientations, 180)

    response_maps = {}
    receptive_fields = {}

    # Compute each oriented receptive field and its response
    for angle in orientations:
        theta = np.deg2rad(angle)

        kernel = gabor(
            sigma=sigma,
            theta=theta,
            Lambda=Lambda,
            psi=psi,
            gamma=gamma
        )

        kernel = _normalize_kernel(kernel)

        receptive_fields[float(angle)] = kernel

        response_maps[float(angle)] = correlate2d(
            image,
            kernel,
            mode="same",
            boundary="symm"
        )

    # Stack signed responses: (orientations, height, width)
    response_stack = np.stack(
        list(response_maps.values()),
        axis=0
    )

    # Strongest response regardless of contrast polarity
    edge_response = np.max(np.abs(response_stack), axis=0)

    # Normalize for visualization and thresholding
    max_response = edge_response.max()

    if max_response > 1e-12:
        edge_response = edge_response / max_response
    else:
        edge_response = np.zeros_like(edge_response)

    # Convert the combined response into a binary edge map
    threshold = np.percentile(
        edge_response,
        threshold_percentile
    )

    binary_response = (
        (edge_response >= threshold) & (edge_response > 0)
    ).astype(np.uint8)

    return (
        response_maps,
        edge_response,
        binary_response,
        receptive_fields
    )

def compute_complex_cells_response(
    image,
    sigma=3.0,
    Lambda=6.0,
    gamma=0.5,
    orientations=None,
    threshold_percentile=90
):
    """
    Compute complex-cell responses using a quadrature pair of Gabor filters.

    For each orientation, simple-cell responses at phases 0 and pi/2
    are combined using the energy model:

        complex_response = sqrt(response_0**2 + response_90**2)

    Parameters
    ----------
    image : ndarray
        2D grayscale input image.
    sigma : float
        Gaussian envelope scale.
    Lambda : float
        Wavelength of the sinusoidal component.
    gamma : float
        Spatial aspect ratio.
    orientations : sequence of float, optional
        Orientations in degrees. Defaults to 0, 22.5, ..., 157.5.
    threshold_percentile : float
        Percentile used to produce the binary response map.

    Returns
    -------
    complex_response_maps : dict
        Complex-cell energy map for each orientation.
    combined_response : ndarray
        Maximum complex-cell response across orientations, normalized
        to [0, 1].
    binary_response : ndarray
        Binary map of strong complex-cell responses.
    """
    image = np.asarray(image, dtype=np.float64)

    if image.ndim != 2:
        raise ValueError("Input image must be a 2D grayscale image.")

    if image.size == 0:
        raise ValueError("Input image cannot be empty.")

    if sigma <= 0 or Lambda <= 0 or gamma <= 0:
        raise ValueError("sigma, Lambda, and gamma must be positive.")

    if not 0 <= threshold_percentile <= 100:
        raise ValueError("threshold_percentile must be between 0 and 100.")

    if orientations is None:
        orientations = np.arange(0, 180, 22.5)

    orientations = np.asarray(orientations, dtype=float)

    if orientations.ndim != 1 or orientations.size == 0:
        raise ValueError("orientations must be a non-empty sequence.")

    if not np.all(np.isfinite(orientations)):
        raise ValueError("orientations must contain finite values.")

    orientations = np.mod(orientations, 180)

    complex_response_maps = {}

    for angle in orientations:
        theta = np.deg2rad(angle)

        # Simple-cell filter 1: phase 0
        kernel_0 = gabor(
            sigma=sigma,
            theta=theta,
            Lambda=Lambda,
            psi=0.0,
            gamma=gamma
        )
        kernel_0 = _normalize_kernel(kernel_0)

        # Simple-cell filter 2: phase pi/2
        kernel_90 = gabor(
            sigma=sigma,
            theta=theta,
            Lambda=Lambda,
            psi=np.pi / 2,
            gamma=gamma
        )
        kernel_90 = _normalize_kernel(kernel_90)

        # Signed simple-cell responses
        response_0 = correlate2d(
            image, kernel_0, mode="same", boundary="symm"
        )

        response_90 = correlate2d(
            image, kernel_90, mode="same", boundary="symm"
        )

        # Complex-cell energy response
        complex_response = np.sqrt(
            response_0**2 + response_90**2
        )

        complex_response_maps[float(angle)] = complex_response

    # Pool responses across orientations
    response_stack = np.stack(
        list(complex_response_maps.values()),
        axis=0
    )

    combined_response = np.max(response_stack, axis=0)

    # Normalize to [0, 1]
    max_response = combined_response.max()

    if max_response > 1e-12:
        combined_response = combined_response / max_response
    else:
        combined_response = np.zeros_like(combined_response)

    # Threshold to produce a binary map
    threshold = np.percentile(
        combined_response, threshold_percentile
    ) 

    binary_response = (
        (combined_response >= threshold) &
        (combined_response > 0)
    ).astype(np.uint8)

    return (
        complex_response_maps,
        combined_response,
        binary_response
    )
