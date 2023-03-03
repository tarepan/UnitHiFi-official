"""©tarepan, licensed to the world under MIT LICENSE."""


def normalize_nonzero(x, mean: float, std: float):
    """Normalize non-zero components with given parameters.

    Args:
        x :: Tensor | NDArray - Target
        mean - External mean parameter
        std  - External std parameter
    """

    idx_non0 = x != 0
    x[idx_non0] = (x[idx_non0] - mean) / std

    return x
