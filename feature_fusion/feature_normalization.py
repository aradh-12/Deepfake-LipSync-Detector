import numpy as np


def normalize_lip_coordinates(coordinates):
    """
    Normalize 40 lip landmarks while preserving temporal
    mouth-shape information.

    Input:
        coordinates: numpy array of shape (40, 2)

    Output:
        normalized coordinates: numpy array of shape (40, 2)

    Method:
        1. Center landmarks around the mouth center.
        2. Use mouth width as ONE common scale factor.
        3. Do NOT independently normalize X and Y.

    This preserves relative vertical mouth movement, which
    is important for lip-sync anomaly detection.
    """

    coordinates = np.asarray(
        coordinates,
        dtype=np.float32
    )

    if coordinates.shape != (40, 2):

        raise ValueError(
            f"Expected coordinates of shape (40, 2), "
            f"got {coordinates.shape}"
        )

    # --------------------------------------------------
    # Mouth center
    # --------------------------------------------------

    center = np.mean(
        coordinates,
        axis=0
    )

    centered = (
        coordinates - center
    )

    # --------------------------------------------------
    # Mouth width
    # --------------------------------------------------

    x_min = np.min(
        coordinates[:, 0]
    )

    x_max = np.max(
        coordinates[:, 0]
    )

    mouth_width = (
        x_max - x_min
    )

    # Prevent division by zero
    if mouth_width < 1e-6:
        mouth_width = 1.0

    # --------------------------------------------------
    # Use ONE scale for both X and Y
    # --------------------------------------------------

    normalized = (
        centered / mouth_width
    )

    return normalized.astype(
        np.float32
    )
