import config
import numpy as np
from scipy.ndimage import convolve, shift
from skimage.feature import canny

def compute_dog_filters(sigma):
  """Computes center-on (DoG+) and center-off (DoG-) difference of Gaussians filters.

  Parameters:
  - sigma: Standard deviation of the outer Gaussian function.
           The inner Gaussian standard deviation is fixed to 0.5 * sigma.
  """
  sigma_inner = 0.5 * sigma
  sigma_outer = sigma

  # Determine kernel size based on the outer sigma (covers 6 * sigma)
  k_size = int(np.ceil(6 * sigma_outer))
  if k_size % 2 == 0:
    k_size += 1  # ensure odd kernel size

  ax = np.arange(-k_size // 2 + 1, k_size // 2 + 1)
  xx, yy = np.meshgrid(ax, ax)
  r2 = xx**2 + yy**2

  # 2D Gaussian functions
  g_inner = (1 / (2 * np.pi * sigma_inner**2)) * np.exp(
      -r2 / (2 * sigma_inner**2)
  )
  g_outer = (1 / (2 * np.pi * sigma_outer**2)) * np.exp(-r2 / (2 * sigma_outer**2))

  # Center-on DoG filter (Excitatory center, inhibitory surround)
  dog_plus = g_inner - g_outer

  # Center-off DoG filter (Inhibitory center, excitatory surround)
  dog_minus = -dog_plus

  return dog_plus, dog_minus


def half_wave_rectification(arr):
  """Applies half-wave rectification: |x|^+ = max(0, x)."""
  return np.maximum(0, arr)


def lgn_cell_response(image, sigma, polarity="+"):
  """Computes the response of model LGN cells across the image.

  Parameters:
  - image: 2D numpy array representing the input image.
  - sigma: Scale parameter for the DoG filter.
  - polarity: '+' for center-on or '-' for center-off.
  """
  dog_plus, dog_minus = compute_dog_filters(sigma)
  kernel = dog_plus if polarity == "+" else dog_minus

  # Linear spatial summation (convolution) followed by half-wave rectification
  conv_result = convolve(image.astype(float), kernel, mode="reflect")
  response = half_wave_rectification(conv_result)

  return response

def compute_subunit_response(image, subunit):
  """Computes the response map of a single sub-unit

  Parameters:
  - image: 2D numpy array
  - subunit: dict with keys 'delta', 'sigma', 'rho', 'phi'
  """
  delta = subunit["delta"]
  sigma = subunit["sigma"]
  rho = subunit["rho"]
  phi = subunit["phi"]

  # Compute base LGN cell response map
  lgn_resp = lgn_cell_response(image, sigma, polarity=delta)

  # Standard deviation sigma' of the weighting Gaussian function G_sigma'
  # sigma' = (d0 + alpha * rho) / 6, with d0 = 2, alpha = 0.9
  d0 = 2.0
  alpha = 0.9
  sigma_prime = (d0 + alpha * rho) / 6.0

  # Create 2D Gaussian kernel G_sigma' (-3*sigma' to 3*sigma')
  k_size = int(np.ceil(6 * sigma_prime))
  if k_size % 2 == 0:
    k_size += 1
  ax = np.arange(-k_size // 2 + 1, k_size // 2 + 1)
  xx, yy = np.meshgrid(ax, ax)
  r2 = xx**2 + yy**2
  g_weight = (1 / (2 * np.pi * sigma_prime**2)) * np.exp(
      -r2 / (2 * sigma_prime**2)
  )

  # Convolve LGN response with weighting function
  convolved = convolve(lgn_resp, g_weight, mode="reflect")

  # Shift vector determined by sub-unit parameters (rho, phi)
  delta_x = -rho * np.cos(phi)
  delta_y = -rho * np.sin(phi)

  # Apply spatial shift (shifting by delta_y, delta_x)
  subunit_resp = shift(
      convolved, shift=(delta_y, delta_x), mode="constant", cval=0.0
  )

  return subunit_resp

def compute_corf_response(image, sub_units):
  """Computes the response r_S(x, y) of a CORF model cell
  as the weighted geometric mean of all sub-unit responses
  """
  h, w = image.shape
  num_units = len(sub_units)

  # Accumulate log responses for numerical stability (geometric mean)
  log_sum = np.zeros((h, w))
  epsilon = 1e-6  # prevent log(0)

  for su in sub_units:
    su_resp = compute_subunit_response(image, su)
    safe_resp = np.clip(su_resp, epsilon, None)
    log_sum += np.log(safe_resp)

  # Geometric mean: exp((1 / N) * sum(log(responses)))
  corf_resp = np.exp(log_sum / num_units)
  return corf_resp

def create_oriented_subunit_set(base_sub_units, psi):
  """Creates a new set of sub-unit parameters R_psi(S) with orientation preference psi

  Parameters:
  - base_sub_units: List of dicts representing sub-units configured for 0 degrees.
  - psi: Orientation preference offset in radians.
  """
  rotated_sub_units = []
  for su in base_sub_units:
    new_su = su.copy()
    # Add psi to the polar angle phi, wrapping within [0, 2*pi]
    new_su["phi"] = (su["phi"] + psi) % (2 * np.pi)
    rotated_sub_units.append(new_su)
  return rotated_sub_units

def compute_max_superposition_response(image, base_sub_units=None, n_theta=None):
  """Computes the maximum superposition response r_hat_S(x, y) across n_theta
  quidistant orientations 

  Parameters:
  - image: 2D numpy array representing the input image.
  - base_sub_units: Base sub-unit configuration set S.
  - n_theta: Number of equidistant orientation preferences (default 12, intervals of pi/6).
  """
  if base_sub_units is None:
    base_sub_units = config.SUBUNIT_PARAMETERS
  if n_theta is None:
    n_theta = config.N_THETA

  h, w = image.shape
  max_response_map = np.zeros((h, w))

  # Define set of equidistant orientations: Psi = { (2*pi / n_theta) * i | 0 <= i < n_theta }
  orientations = [(2 * np.pi / n_theta) * i for i in range(n_theta)]

  for psi in orientations:
    # Generate sub-unit set for orientation preference psi
    psi_sub_units = create_oriented_subunit_set(base_sub_units, psi)

    # Compute CORF response for this specific orientation
    resp_map = compute_corf_response(image, psi_sub_units)

    # Take the maximum response pixel-wise across all evaluated orientations
    max_response_map = np.maximum(max_response_map, resp_map)

  return max_response_map

def generate_corf_contour_map(image, sub_units=None, n_theta=None):
  """Computes the maximum superposition CORF contour map for an input image

  and produces a clean binary contour map (matching Figure 12).
  """
  if sub_units is None:
    sub_units = config.SUBUNIT_PARAMETERS
  if n_theta is None:
    n_theta = config.N_THETA

  h, w = image.shape
  max_superposition = np.zeros((h, w))

  # Compute equidistant orientations Psi (intervals of pi/6 for 12 orientations)
  orientations = [(2 * np.pi / n_theta) * i for i in range(n_theta)]

  for psi in orientations:
    # Rotate sub-unit set by psi
    rotated_su = create_oriented_subunit_set(sub_units, psi)
    # Compute CORF response map
    resp_map = compute_corf_response(image, rotated_su)
    # Accumulate pixel-wise maximum superposition
    max_superposition = np.maximum(max_superposition, resp_map)

  # Normalize response map between 0 and 1
  if max_superposition.max() > 0:
    max_superposition /= max_superposition.max()

  # Binarization & Thinning
  low_thresh = 0.15 * max_superposition.max()
  high_thresh = 0.3 * max_superposition.max()

  binary_contours = canny(
      max_superposition, sigma=1.0, low_threshold=low_thresh, high_threshold=high_thresh
  )

  return max_superposition, binary_contours