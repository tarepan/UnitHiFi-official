"""©tarepan, licensed to the world under MIT LICENSE."""

import numpy as np

from preprocess import normalize_nonzero


def test_normalize_nonzero_numpy():
    mean, std = 2, 3
    ipt = np.array([1.,     2., 4.,])
    opt = np.array([-1./3., 0., +2./3.,])

    np.testing.assert_almost_equal(opt, normalize_nonzero(ipt, mean, std))

# def test_normalize_nonzero_pytorch():
#     import torch

#     mean, std = 2, 3
#     ipt = torch.tensor([1.,     2., 4.,])
#     opt = torch.tensor([-1./3., 0., +2./3.,])

#     assert torch.allclose(opt, normalize_nonzero(ipt, mean, std))
#     # Test passed @ 2023-03-03T11:00+09:00
