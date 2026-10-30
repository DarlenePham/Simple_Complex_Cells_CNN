import numpy as np
from scipy.ndimage import convolve

def compute_simple_cell_response(image, size=3, threshold_percentile=90):
    """Computes the response of simple cells across the image.

    Parameters
    ----------
    image : 2D numpy array
        Input grayscale image.

    size : int, optional
        Size of the simple cell receptive field. Default is 3.

    threshold_percentile : int, optional
        Percentile for binarization threshold. Default is 90.

    Returns
    -------
    simple_response : 2D numpy array
        Response map of simple cells.
    """
    # Vertical edge detector kernel
    kernel_vertical = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])

    # Horizontal edge detector kernel
    kernel_horizontal = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

    # Convolve the image with the kernel to get the response
    simple_response_vertical = convolve(image, kernel_vertical, mode="reflect")
    simple_response_horizontal = convolve(image, kernel_horizontal, mode="reflect")
    simple_response = np.sqrt(
        simple_response_vertical**2 + simple_response_horizontal**2
    )

    if not (0 <= threshold_percentile <= 100):
        raise ValueError("threshold_percentile must be between 0 and 100")

    # Convert the combined response into a binary edge map using percentile
    threshold = np.percentile(simple_response, threshold_percentile)
    binarized_response = (simple_response > threshold).astype(np.float32)

    return simple_response, binarized_response
